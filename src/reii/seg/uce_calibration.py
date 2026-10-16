"""Auto-calibración bayesiana del tamaño objetivo de UCE (``uce_target_size``).

Contexto
--------
``_segmentar_reinert_spacy`` corta en la mejor puntuación dentro de una
ventana ``[target - 0.4·target, target + 0.4·target]``. Empíricamente, la
puntuación tiende a aparecer antes del centro de esa ventana, así que el
promedio REAL de palabras por UCE no coincide con ``target``: con
``target=26``, REII produce en la práctica un promedio menor (~21.5,
subpasa el objetivo), mientras que ALCESTE original SOBREPASA el suyo
(26 → ~38 palabras/UCE), porque concatena Segments de Texte Calibrés
completos hasta la puntuación natural. El sesgo depende del corpus
(densidad de puntuación, longitud de frase), así que no existe una
constante de corrección universal.

Este módulo resuelve el problema inverso: dado un promedio real deseado,
¿qué ``target`` hay que pedirle al cortador para obtenerlo EN ESTE corpus?
Se modela la relación ``target → promedio_real`` como aproximadamente
lineal en el rango permitido (``real ≈ a + b·target``) y se infieren
``(a, b)`` con regresión lineal bayesiana (prior Normal conjugado,
verosimilitud Normal con varianza heterocedástica por observación —
el error estándar de un promedio decrece con ``1/√n``), refinando con
pocas mediciones secuenciales: cada medición requiere correr spaCy sobre
una muestra del corpus, así que el presupuesto de mediciones es chico
por diseño (5 por defecto).

Nada en este módulo importa spaCy/stanza ni nada del resto de REII: recibe
un ``measure_fn`` que mide (el llamador decide cómo, ver
``SegmentadorALCESTE.medir_promedio_real`` en ``main_workflow.py``). Así
toda la lógica estadística se puede probar con mediciones sintéticas,
deterministas o ruidosas, sin spaCy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

# Rango duro: el objetivo nunca sale de aquí, sin importar qué pida la
# calibración ni la tabla de ALCESTE por tamaño de corpus (ver
# ``desired_avg_for_corpus``). "Rango Reinert" ya citado en el código.
DEFAULT_BOUNDS: Tuple[int, int] = (14, 30)

# Convención de ALCESTE para corpus chico (~20-100 KB): 20-30 ocurrencias,
# con 26 como ejemplo típico citado en su documentación.
DEFAULT_DESIRED_AVG: float = 26.0

# Piso de desviación estándar (en palabras) para evitar que una muestra
# minúscula o perfectamente uniforme colapse la varianza a ~0 y, con ella,
# la incertidumbre de la medición a infinito-precisión espuria.
MIN_STD_FLOOR: float = 1.0


# ─────────────────────────────────────────────────────────────────────────
# Observaciones y posterior
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Observation:
    """Una medición real: al pedir ``target``, el corpus (o una muestra)
    produjo UCEs con este promedio y desviación estándar de palabras."""

    target: int
    real_avg: float
    std_avg: float
    n_uces: int

    def __post_init__(self) -> None:
        if self.n_uces < 1:
            raise ValueError(f"n_uces debe ser >= 1, recibido {self.n_uces}")
        if self.std_avg < 0:
            raise ValueError(f"std_avg no puede ser negativo: {self.std_avg}")

    @property
    def variance_of_mean(self) -> float:
        """Varianza del promedio medido: max(std, piso)² / n — el error
        estándar de una media decrece con 1/√n."""
        std = max(self.std_avg, MIN_STD_FLOOR)
        return (std * std) / self.n_uces


@dataclass(frozen=True)
class BayesianLinearPosterior:
    """Posterior Normal sobre ``(a, b)`` en ``real_avg ≈ a + b·target``."""

    a_mean: float
    b_mean: float
    cov: np.ndarray  # 2x2: [[Var(a), Cov(a,b)], [Cov(a,b), Var(b)]]
    n_obs: int

    def predict(self, target: float) -> float:
        return self.a_mean + self.b_mean * target

    def predict_var(self, target: float) -> float:
        """Varianza de la predicción puntual (propaga la incertidumbre de a, b)."""
        x = np.array([1.0, float(target)])
        return float(x @ self.cov @ x)

    def predict_std(self, target: float) -> float:
        return math.sqrt(max(0.0, self.predict_var(target)))

    def solve_for(self, desired: float, bounds: Tuple[int, int]) -> int:
        """Entero en ``bounds`` cuya predicción está más cerca de ``desired``.

        Si la pendiente estimada es ~0 (el target no mueve el promedio real
        según lo observado hasta ahora — degenerado o aún sin datos
        suficientes), no hay nada que resolver: se devuelve el punto medio
        del rango como propuesta neutral, y el llamador decide qué hacer
        con la falta de convergencia.
        """
        lo, hi = bounds
        if abs(self.b_mean) < 1e-9:
            return int(round((lo + hi) / 2.0))
        raw = (desired - self.a_mean) / self.b_mean
        clipped = min(float(hi), max(float(lo), raw))
        return int(round(clipped))


def fit_bayesian_linear(
    observations: List[Observation],
    prior_a: Tuple[float, float] = (0.0, 1.0e4),
    prior_b: Tuple[float, float] = (1.0, 4.0),
) -> BayesianLinearPosterior:
    """Regresión lineal bayesiana ponderada: ``real_avg ~ a + b·target``.

    prior_a, prior_b : (media, varianza) del prior Normal independiente de
      cada coeficiente. El prior de ``b`` está centrado en 1.0 (sin sesgo
      sistemático) pero con varianza suficiente (desvío 2.0) para que un
      puñado de observaciones reales domine rápido — ni REII (b≈0.83) ni
      ALCESTE (b≈1.46), los dos casos conocidos, están a más de 1 desvío.

    Con 0 observaciones devuelve el prior puro. Con 1 observación, el
    posterior de ``b`` queda prácticamente sin mover (no hay información
    de pendiente en un solo punto) y el de ``a`` se ajusta parcialmente
    hacia el dato, ponderado por su precisión relativa al prior.
    """
    mu0 = np.array([prior_a[0], prior_b[0]])
    Sigma0_inv = np.diag([1.0 / prior_a[1], 1.0 / prior_b[1]])

    if not observations:
        return BayesianLinearPosterior(
            a_mean=mu0[0], b_mean=mu0[1], cov=np.linalg.inv(Sigma0_inv), n_obs=0
        )

    X = np.array([[1.0, float(o.target)] for o in observations])
    y = np.array([o.real_avg for o in observations])
    precisions = np.array([1.0 / o.variance_of_mean for o in observations])
    W = np.diag(precisions)

    Lambda_n = Sigma0_inv + X.T @ W @ X
    Sigma_n = np.linalg.inv(Lambda_n)
    mu_n = Sigma_n @ (Sigma0_inv @ mu0 + X.T @ W @ y)

    return BayesianLinearPosterior(
        a_mean=float(mu_n[0]), b_mean=float(mu_n[1]), cov=Sigma_n, n_obs=len(observations)
    )


# ─────────────────────────────────────────────────────────────────────────
# Calibración secuencial
# ─────────────────────────────────────────────────────────────────────────

# target -> (promedio_real, desvío_std, n_uces_medidas)
MeasureFn = Callable[[int], Tuple[float, float, int]]


@dataclass(frozen=True)
class CalibrationResult:
    target: int
    desired_avg: float
    bounds: Tuple[int, int]
    measured_avg: float
    predicted_avg: float
    converged: bool
    observations: Tuple[Observation, ...]
    posterior: BayesianLinearPosterior

    def summary(self) -> str:
        tag = "convergió" if self.converged else "NO convergió (uso el mejor punto medido)"
        return (
            f"uce_target_size={self.target} · promedio real medido="
            f"{self.measured_avg:.1f} (deseado={self.desired_avg:.1f}) · {tag} · "
            f"{len(self.observations)} mediciones en {self.bounds}"
        )


def _measure(fn: MeasureFn, target: int) -> Observation:
    mean, std, n = fn(target)
    return Observation(target=target, real_avg=float(mean), std_avg=float(std), n_uces=int(n))


def calibrate(
    measure_fn: MeasureFn,
    desired_avg: float = DEFAULT_DESIRED_AVG,
    bounds: Tuple[int, int] = DEFAULT_BOUNDS,
    max_evals: int = 5,
    tol: float = 1.0,
) -> CalibrationResult:
    """Encuentra, dentro de ``bounds``, el ``uce_target_size`` entero cuyo
    promedio real medido está más cerca de ``desired_avg``.

    Procedimiento (bayesiano secuencial, diseñado para pocas mediciones
    porque cada una corre spaCy sobre una muestra del corpus):

      1. Ancla la pendiente midiendo en los DOS EXTREMOS de ``bounds``
         (diseño óptimo para estimar una recta con 2 puntos: maximizar la
         separación en x). Si cualquiera de los extremos ya cumple
         ``tol``, termina ahí sin gastar más mediciones.
      2. Ajusta la regresión bayesiana con las observaciones acumuladas,
         propone el entero más prometedor dentro de ``bounds`` y lo mide.
         Si la propuesta coincide con un target ya medido, el modelo no
         tiene nada nuevo que ofrecer y se detiene (evita medir dos veces
         el mismo punto en un bucle).
      3. Se detiene al alcanzar ``max_evals`` mediciones o ``tol``.

    Devuelve el MEJOR PUNTO REALMENTE MEDIDO (no la predicción del
    modelo): el objetivo final es el resultado medido, la regresión es
    solo la guía para decidir dónde medir a continuación.

    ``measure_fn`` puede lanzar excepciones (p. ej. spaCy no disponible);
    no se capturan aquí — es responsabilidad del llamador decidir el
    fallback (ver ``calibrar_uce_target_size`` en main_workflow.py).
    """
    lo, hi = bounds
    if lo > hi:
        raise ValueError(f"bounds inválido: {bounds} (lo debe ser <= hi)")
    if max_evals < 1:
        raise ValueError("max_evals debe ser >= 1")

    if lo == hi:
        obs = (_measure(measure_fn, lo),)
        posterior = fit_bayesian_linear(list(obs))
        return CalibrationResult(
            target=lo,
            desired_avg=desired_avg,
            bounds=bounds,
            measured_avg=obs[0].real_avg,
            predicted_avg=obs[0].real_avg,
            converged=abs(obs[0].real_avg - desired_avg) <= tol,
            observations=obs,
            posterior=posterior,
        )

    if max_evals < 2:
        # Con un único punto permitido no hay pendiente que estimar: se
        # mide en el extremo más cercano a lo que pide ALCESTE para este
        # tamaño de corpus (heurística simple, sin regresión).
        target = lo if abs(lo - desired_avg) <= abs(hi - desired_avg) else hi
        obs = (_measure(measure_fn, target),)
        posterior = fit_bayesian_linear(list(obs))
        return CalibrationResult(
            target=target,
            desired_avg=desired_avg,
            bounds=bounds,
            measured_avg=obs[0].real_avg,
            predicted_avg=obs[0].real_avg,
            converged=abs(obs[0].real_avg - desired_avg) <= tol,
            observations=obs,
            posterior=posterior,
        )

    observations: List[Observation] = []
    for t in (lo, hi):
        observations.append(_measure(measure_fn, t))
        if abs(observations[-1].real_avg - desired_avg) <= tol:
            posterior = fit_bayesian_linear(observations)
            best = observations[-1]
            return CalibrationResult(
                target=best.target,
                desired_avg=desired_avg,
                bounds=bounds,
                measured_avg=best.real_avg,
                predicted_avg=posterior.predict(best.target),
                converged=True,
                observations=tuple(observations),
                posterior=posterior,
            )

    for _ in range(max_evals - 2):
        posterior = fit_bayesian_linear(observations)
        candidate = posterior.solve_for(desired_avg, bounds)
        if any(o.target == candidate for o in observations):
            break
        observations.append(_measure(measure_fn, candidate))
        if abs(observations[-1].real_avg - desired_avg) <= tol:
            break

    posterior = fit_bayesian_linear(observations)
    best = min(observations, key=lambda o: abs(o.real_avg - desired_avg))
    return CalibrationResult(
        target=best.target,
        desired_avg=desired_avg,
        bounds=bounds,
        measured_avg=best.real_avg,
        predicted_avg=posterior.predict(best.target),
        converged=abs(best.real_avg - desired_avg) <= tol,
        observations=tuple(observations),
        posterior=posterior,
    )


# ─────────────────────────────────────────────────────────────────────────
# Objetivo deseado por tamaño de corpus (tabla de ALCESTE, acotada)
# ─────────────────────────────────────────────────────────────────────────


def desired_avg_for_corpus(total_words: int) -> float:
    """Promedio real objetivo según la convención de auto-calibración de
    ALCESTE por volumen de corpus (ocurrencias por UCE):

      · chico   (<~25,000 palabras, ~20-100 KB) → 26
      · mediano (<~300,000 palabras, varios MB) → 55
      · masivo  (>=300,000 palabras)            → 120

    Esta función NO decide ``uce_target_size`` directamente — devuelve el
    PROMEDIO REAL deseado, que ``calibrate()`` bounded a ``[14, 30]``
    luego traduce al ``target`` de entrada que realmente lo produce en
    este corpus. Para corpus "mediano" o "masivo", el objetivo (55 o 120)
    queda fuera de ese rango a propósito: la calibración se acerca lo más
    posible desde adentro del rango permitido, sin pretender igualar
    ALCESTE en corpus grandes (fuera de alcance para datos de entrevistas
    cualitativas, que es el uso real de REII).
    """
    if total_words < 25_000:
        return 26.0
    if total_words < 300_000:
        return 55.0
    return 120.0
