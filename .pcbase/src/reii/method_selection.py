"""Selección automática y persistida de UN método de clasificación (A, B o C).

Contexto
────────
El pipeline tenía modos de *intersección* (``all``, ``two_of_three``,
``wc_coref``...): una UCE era "estable" solo si dos o tres métodos coincidían.
Eso no es lo que hace ALCESTE 2012 —que usa una sola doble clasificación por
recuento de palabras— y además no es identificable: con la misma segmentación
el número de clases K salta entre 3, 7 y 16 según los hiperparámetros del
clasificador.

Este módulo implementa la decisión contraria: **elegir exactamente un método**
entre A (word-count), B (similitud/coref) y C (embeddings/HDBSCAN), con una
regla **pre-registrada** (fijada antes de ver los resultados), y **persistir**
la elección junto al corpus al que se aplicó. Nada de intersecciones.

Regla (lexicográfica, versión ``select-v4``)
────────────────────────────────────────────
1. **Viabilidad**: el método existe (produjo partición), **no es diagnóstico**
   (``diagnostic_only`` o ``backend="hdbscan"``: no controla K y su ARI alto es
   artefacto del colapso, ``BACKEND_C.md`` §2), su ``K`` cae en ``k_bounds``
   (3..20: ``K=2`` es trivial y su ARI=1.0 no significa nada), su **ruido** no
   supera ``noise_max`` (0.50) y su **cobertura efectiva**
   (``coverage_stable`` si el motor la produjo; si no ``coverage``) alcanza
   ``coverage_floor`` (0.15).
2. **No dominancia sobre el corpus**: ``c = a * (1 - nu)``, con
   ``a = max_class_share`` y ``nu = noise_frac``. Si ``c > C0`` (0.70) la
   partición es degenerada (una clase + satélites) y el método se descarta.
   Medir sobre lo asignado hacía depender el veredicto del ruido, y con el
   signo equivocado: misma ``c = 0.837``, aceptada sin ruido y descartada con
   ruido. Derivación y calibración en ``docs/design_v5/GATE_NOISE.md``.
3. **Contraste contra el nulo** (``null_p``): ``null_mode`` decide qué se hace
   con él. ``"gate"`` (default) descarta si ``null_p > null_alpha``; ``"rank"``
   no descarta y antepone la clase de evidencia al orden; ``"off"`` lo ignora.
   ``nan`` = no disponible y **nunca** descarta. Con un nulo de etiquetas
   permutadas el gate es débil (no puede rechazar ruido); con un nulo de datos
   es el que produce abstención sobre ruido. Ver ``SELECT_V4.md`` §4.
4. **Criterio principal: estabilidad**. ``stability = bootstrap_ari`` si existe
   (remuestreo real: submuestra + reajuste), si no ``ari`` (pase tight/loose).
   El criterio usado se declara en ``winner_stability_kind``.
5. **Las familias no son la misma escala**: un ARI de bootstrap y un ARI de
   perturbación no son comparables. La regla ordena igual (no hay escala común
   disponible) pero si ganador y runner-up usan kinds distintos marca
   ``criteria_conflict`` y lo explica. Además, ``backend in
   {"ward","kmeans_spherical"}`` no tiene grados de libertad en el pase
   tight/loose (``ARI == 1.0`` por construcción, ``BACKEND_C.md`` §3): ese
   ``ari`` es **tautológico** y, sin ``bootstrap_ari``, no compite
   (``reject_tautological``).
6. **Desempate**: coverage efectiva (``usable_effective``) y, por último, orden
   canónico ``A < B < C``.
7. **Ambigüedad declarada**: si ``|dStab| <= margin`` la elección se marca
   ``ambiguous``. Si además el runner-up cubre más, o los kinds difieren, se
   marca ``criteria_conflict``. El ganador se mantiene —la regla es
   determinista— pero el run queda marcado para revisión humana.
8. **Abstención**: si ningún método es viable, ``winner=None`` y
   ``abstained=True``. El pipeline **no clasifica** pero continúa: las UCEs
   quedan sin ``cluster_id`` y las etapas posteriores corren sobre vacío.
   Nunca se lanza una excepción por métricas incompletas: todo campo nuevo cae
   a ``nan`` = "no disponible" y su señal no interviene.

Precisión de las medidas (todas con denominador ``n_total`` de UCEs)
───────────────────────────────────────────────────────────────
* ``coverage``   = UCEs estables del método (misma etiqueta en tight y loose)
  / ``n_total``.
* ``noise_frac`` = UCEs **sin asignar** (``label < 0``, p.ej. HDBSCAN ``-1``)
  / ``n_total``. Antes se dividía por el número de etiquetas del propio método
  y por eso A y B daban 0.00 por construcción (el CDH nunca deja ``-1``):
  ahora es comparable entre métodos.
* ``tiny_frac``  = UCEs **estables en clases diminutas** (< ``min_class_share``
  del corpus, 1 % por defecto) / ``n_total``. Captura el fallo real de A/B:
  clases de 1-2 UCEs que son "estables" pero no interpretables. Sin esto, un
  K inflado con clases basura parecía impecable.
* ``usable``     = ``coverage - tiny_frac`` (UCEs estables **y** en una clase
  con masa suficiente). Es la cantidad que se usa para el piso y el desempate.

Las perturbaciones intra-método NO son de la misma naturaleza (A mueve
``min_forms_uc``, B mueve la ventana de coref, C mueve HDBSCAN): por eso cada
``MethodMetrics`` lleva su ``perturbation`` y el ARI se compara solo dentro de
su propio método, nunca entre métodos.

Uso
───
    from reii.method_selection import MethodMetrics, select_method

    sel = select_method([
        MethodMetrics("A", k=7, ari=0.419, coverage=0.394, perturbation="A1<->A2"),
        MethodMetrics("B", k=9, ari=0.212, coverage=0.415, perturbation="B1<->B2"),
    ])
    print(sel.explain_text())
    persist_selection(sel, "data/method_selection.json",
                      history_path="data/method_selection_log.jsonl",
                      context={"corpus_fingerprint": "abc123", "target": 26})
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RULE_VERSION = "select-v4"

#: Piso de viabilidad sobre el ARI bootstrap. Medido: ruido 0.09-0.11,
#: corpus real 0.42-0.60, estructura 0.78-0.97 (STABILITY_NULL.md §2-§3).
#: Debe coincidir con ``quality.metrics.STABILITY_MEAN_MIN``.
STABILITY_MEAN_MIN = 0.30

#: Etiquetas legibles por método. Se usan solo para imprimir.
METHOD_LABELS: Dict[str, str] = {
    "A": "word-count (ALCESTE A1/A2)",
    "B": "similitud/coref (B1/B2)",
    "C": "embeddings/HDBSCAN (C1/C2)",
}

#: Orden canónico de desempate final (determinista).
_METHOD_ORDER = {"A": 0, "B": 1, "C": 2}


# ══════════════════════════════════════════════════════════════════════════
# Estructuras
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MethodMetrics:
    """Medidas de UN método dentro de una misma corrida.

    ``ari``        : acuerdo entre los dos pases del método (tight vs loose).
    ``coverage``   : UCEs estables del método / ``n_total``.
    ``noise_frac`` : UCEs sin asignar (p.ej. HDBSCAN ``-1``) / ``n_total``.
    ``tiny_frac``  : UCEs estables en clases < ``min_class_share`` / ``n_total``.
    ``available``  : ``False`` si el método no produjo partición utilizable.
    """

    method: str
    k: int
    ari: float
    coverage: float
    noise_frac: float = 0.0
    tiny_frac: float = 0.0
    perturbation: str = ""
    available: bool = True
    # ── diagnósticos (no entran en la decisión) ──────────────────────────
    assigned_frac: Optional[float] = None
    min_class_size: int = 0
    median_class_size: float = 0.0
    #: Tamaño de la clase mayor / UCEs asignadas (partición cruda del método).
    #: Alimenta el gate de no dominancia; también se imprime.
    max_class_share: float = 0.0
    # ── select-v4: señales nuevas (default = "no disponible") ────────────
    #: "" (A/B) | "hdbscan" | "dense_cdh" | "ward" | "kmeans_spherical".
    backend: str = ""
    #: True => el método es diagnóstico y no compite (C con hdbscan).
    diagnostic_only: bool = False
    #: Estabilidad por remuestreo (ARI medio contra la partición completa).
    bootstrap_ari: float = float("nan")
    bootstrap_ci_lo: float = float("nan")
    bootstrap_ci_hi: float = float("nan")
    #: Cobertura con estabilidad por UCE (nan => se usa ``coverage``).
    coverage_stable: float = float("nan")
    #: p-valor contra etiquetas permutadas (nan => no disponible).
    null_p: float = float("nan")
    null_z: float = float("nan")
    n_total: int = 0
    n_assigned: int = 0
    n_noise: int = 0
    max_class_size: int = 0
    #: c = a * (1 - nu) medido por el productor; nan => se deriva.
    dominance_corpus: float = float("nan")

    @property
    def usable_coverage(self) -> float:
        """UCEs estables y en una clase con masa suficiente / ``n_total``."""
        return max(0.0, float(self.coverage) - float(self.tiny_frac))

    @property
    def coverage_effective(self) -> float:
        """``coverage_stable`` si el motor la produjo; si no ``coverage``.

        En un backend denso ``coverage`` es 1.0 por construcción (asigna todo),
        así que premiarla premiaría a C siempre. ``coverage_stable`` mide la
        fracción de UCEs con etiqueta estable en el remuestreo.
        """
        stable = float(self.coverage_stable)
        return stable if math.isfinite(stable) else float(self.coverage)

    @property
    def usable_effective(self) -> float:
        """Cobertura efectiva descontando las clases diminutas."""
        return max(0.0, self.coverage_effective - float(self.tiny_frac))

    @property
    def dominance_corpus_effective(self) -> float:
        """Clase mayor / ``n_total``, con precedencia explícita.

        ``1`` si el productor midió ``dominance_corpus``; ``2`` el cociente
        exacto ``max_class_size / n_total``; ``3`` la identidad
        ``max_class_share * (1 - noise_frac)`` (``GATE_NOISE.md`` §1). Los
        conteos van antes que la identidad porque si ``a`` se calculó mal,
        ``s_max / n_total`` lo delata.
        """
        measured = float(self.dominance_corpus)
        if math.isfinite(measured):
            return measured
        if int(self.n_total) > 0 and int(self.max_class_size) > 0:
            return float(self.max_class_size) / float(self.n_total)
        share = float(self.max_class_share or 0.0)
        noise = min(1.0, max(0.0, float(self.noise_frac or 0.0)))
        return share * (1.0 - noise)

    @property
    def stability(self) -> float:
        """Criterio principal: bootstrap si existe, perturbación si no."""
        boot = float(self.bootstrap_ari)
        return boot if math.isfinite(boot) else float(self.ari)

    @property
    def stability_kind(self) -> str:
        """``"bootstrap"`` | ``"perturbation"`` (lo que mide ``stability``)."""
        return "bootstrap" if math.isfinite(float(self.bootstrap_ari)) else "perturbation"

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["usable"] = self.usable_coverage
        row["usable_eff"] = self.usable_effective
        row["coverage_effective"] = self.coverage_effective
        row["dominance_corpus_eff"] = self.dominance_corpus_effective
        row["stability"] = self.stability
        row["stability_kind"] = self.stability_kind
        return row


@dataclass
class Selection:
    """Resultado de la regla. Serializable y persistible."""

    winner: Optional[str]
    winner_ari: float
    winner_coverage: float
    winner_k: int
    winner_usable_coverage: float = float("nan")
    runner_up: Optional[str] = None
    runner_up_ari: float = float("nan")
    runner_up_coverage: float = float("nan")
    runner_up_k: int = 0
    runner_up_usable_coverage: float = float("nan")
    delta_ari: float = float("nan")
    delta_coverage: float = float("nan")
    ambiguous: bool = False
    criteria_conflict: bool = False
    abstained: bool = False
    reason: str = ""
    discarded: List[Dict[str, Any]] = field(default_factory=list)
    all_metrics: List[Dict[str, Any]] = field(default_factory=list)
    rule_version: str = RULE_VERSION
    margin: float = 0.02
    coverage_floor: float = 0.15
    min_class_share: float = 0.01
    noise_max: float = 0.50
    #: Gate de dominancia sobre el CORPUS: c = max_class_share*(1-noise_frac).
    max_class_share_max: float = 0.70
    k_bounds: Tuple[int, int] = (3, 20)
    # ── select-v4 ────────────────────────────────────────────────────────
    #: Escala del umbral de dominancia: la clase mayor se mide sobre el corpus.
    dominance_scale: str = "corpus"
    #: "diagnostic" (default) = columna de control; "gate" solo con nulo de
    #: datos; "rank" y "off" se conservan. Ver STABILITY_NULL.md §7b.
    null_mode: str = "diagnostic"
    #: Nivel del contraste contra el nulo (p de una cola).
    null_alpha: float = 0.05
    #: Estabilidad del ganador y el criterio con el que se midió.
    winner_stability: float = float("nan")
    winner_stability_kind: str = ""
    winner_null_p: float = float("nan")
    #: c del ganador (clase mayor / n_total). Para el auditor, no para decidir.
    winner_dominance_corpus: float = float("nan")
    runner_up_stability: float = float("nan")
    runner_up_stability_kind: str = ""

    # ── serialización ───────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["k_bounds"] = list(self.k_bounds)
        # NaN no es JSON válido: se emite como null.
        for key, value in list(d.items()):
            if isinstance(value, float) and not math.isfinite(value):
                d[key] = None
        d["all_metrics"] = [
            {
                k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in row.items()
            }
            for row in d["all_metrics"]
        ]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Selection":
        d = dict(d)
        d["k_bounds"] = tuple(d.get("k_bounds") or (3, 20))
        for key in (
            "winner_ari",
            "winner_coverage",
            "winner_usable_coverage",
            "runner_up_ari",
            "runner_up_coverage",
            "runner_up_usable_coverage",
            "delta_ari",
            "delta_coverage",
            "winner_stability",
            "winner_null_p",
            "winner_dominance_corpus",
            "runner_up_stability",
        ):
            if d.get(key) is None:
                d[key] = float("nan")
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})

    # ── presentación (ASCII puro: la consola del host es cp1252) ─────────
    def explain_lines(self) -> List[str]:
        out = [
            f"METHOD SELECTION (rule {self.rule_version})",
            f"  dominancia sobre '{self.dominance_scale}': c=a*(1-nu) > "
            f"{self.max_class_share_max:.2f} | nulo={self.null_mode}"
            f"@{self.null_alpha:.2f} | margen {self.margin:.2f}",
            f"  {'met':4}{'K':>4}{'stab':>7}{'knd':>6}{'cov':>7}{'covS':>7}"
            f"{'usable':>8}{'noise':>7}{'dom':>6}{'domC':>7}{'nullp':>7}  "
            f"{'perturbacion':14}  veredicto",
        ]
        for row in self.all_metrics:
            tag = ""
            if row["method"] == self.winner:
                tag = "<- winner"
            elif row["method"] == self.runner_up:
                tag = "<- runner-up"
            if not row.get("available", True):
                tag = "no disponible"
            out.append(
                f"  {row['method']:4}{int(row.get('k') or 0):>4}"
                f"{_fmt(row.get('stability')):>7}"
                f"{str(row.get('stability_kind') or '-'):>6}"
                f"{float(row.get('coverage') or 0.0):>7.3f}"
                f"{_fmt(row.get('coverage_stable')):>7}"
                f"{float(row.get('usable') or 0.0):>8.3f}"
                f"{float(row.get('noise_frac') or 0.0):>7.2f}"
                f"{float(row.get('max_class_share') or 0.0):>6.2f}"
                f"{_fmt(row.get('dominance_corpus_eff')):>7}"
                f"{_fmt(row.get('null_p')):>7}  "
                f"{str(row.get('perturbation') or '-'):14}  {tag}"
            )
        for item in self.discarded:
            out.append(f"  [!] {item['method']} descartado: {item['why']}")
        if self.winner is None:
            out.append(
                "  [X] SIN CLASIFICACION (abstencion): "
                f"{self.reason}. El pipeline continua sin clases."
            )
            return out
        out.append(
            f"  => metodo elegido: {self.winner} "
            f"({METHOD_LABELS.get(self.winner, '?')})"
        )
        out.append(
            f"     estabilidad={_fmt(self.winner_stability, '.3f')} "
            f"({self.winner_stability_kind or '-'}; ARI={self.winner_ari:.3f})  "
            f"coverage={self.winner_coverage:.3f}  "
            f"usable={self.winner_usable_coverage:.3f}  K={self.winner_k}"
        )
        out.append(
            f"     nulo: p={_fmt(self.winner_null_p, '.3f')}"
        )
        _win_row = next((r for r in self.all_metrics if r["method"] == self.winner), {})
        out.append(
            f"     clase mayor={float(_win_row.get('max_class_share') or 0.0):.2f} "
            f"de las asignadas = {_fmt(_win_row.get('dominance_corpus_eff'))} del corpus "
            f"(gate: >{self.max_class_share_max:.2f} del corpus = "
            "particion degenerada)"
        )
        out.append(
            f"     control: bootstrap={_fmt(_win_row.get('bootstrap_ari'))}"
            f" IC95=[{_fmt(_win_row.get('bootstrap_ci_lo'))},"
            f"{_fmt(_win_row.get('bootstrap_ci_hi'))}]"
            f" nulo p={_fmt(_win_row.get('null_p'), '.3f')}"
            f" (modo {self.null_mode}"
            f"{' - filtra' if self.null_mode == 'gate' else ' - no filtra'})"
        )
        if self.ambiguous and self.runner_up:
            out.append(
                f"  [!] AMBIGUO: {self.winner} vs {self.runner_up}, "
                f"|dStab|={abs(self.delta_ari):.3f} <= margin={self.margin:.3f}"
            )
        if self.criteria_conflict and self.runner_up:
            if self.winner_stability_kind != self.runner_up_stability_kind:
                out.append(
                    "  [!] conflicto de criterios: "
                    f"{self.winner} mide '{self.winner_stability_kind}' y "
                    f"{self.runner_up} mide '{self.runner_up_stability_kind}'; "
                    "no estan en la misma escala (bootstrap = remuestreo, "
                    "perturbacion = pase tight/loose)"
                )
            if (
                math.isfinite(float(self.runner_up_usable_coverage))
                and self.runner_up_usable_coverage
                > self.winner_usable_coverage + 1e-12
            ):
                out.append(
                    f"  [!] conflicto de criterios: {self.runner_up} tiene mas "
                    f"coverage util ({self.runner_up_usable_coverage:.3f} vs "
                    f"{self.winner_usable_coverage:.3f}); la estabilidad manda "
                    "pero la eleccion no es holgada"
                )
        return out

    def explain_text(self) -> str:
        return "\n".join(self.explain_lines())


# ══════════════════════════════════════════════════════════════════════════
# Regla
# ══════════════════════════════════════════════════════════════════════════
#: Backends que son solo diagnóstico: no compiten (``BACKEND_C.md`` §2).
_DIAGNOSTIC_BACKENDS: Tuple[str, ...] = ("hdbscan",)

#: Backends sin grados de libertad en el pase tight/loose: ``ari == 1.0`` por
#: construcción (``BACKEND_C.md`` §3), así que ese ARI no mide estabilidad.
_TAUTOLOGICAL_BACKENDS: Tuple[str, ...] = ("ward", "kmeans_spherical")

#: Modos de uso del contraste contra el nulo. "diagnostic" persiste e imprime
#: el p-valor pero no descarta ni ordena (STABILITY_NULL.md §1.3).
_NULL_MODES: Tuple[str, ...] = ("gate", "rank", "off", "diagnostic")


def _fmt(value: Any, spec: str = ".2f") -> str:
    """Formatea un float que puede ser ``NaN`` sin romper la tabla."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    return "nan" if not math.isfinite(number) else format(number, spec)


def _null_class(metric: MethodMetrics, null_alpha: float) -> int:
    """``0`` = evidencia contra el nulo, ``1`` = sin medir, ``2`` = sin evidencia.

    Se usa solo con ``null_mode="rank"``; el gate compara el p-valor directo.
    """
    p = float(metric.null_p)
    if not math.isfinite(p):
        return 1
    return 0 if p <= float(null_alpha) else 2


def _feasibility(
    metric: MethodMetrics,
    coverage_floor: float,
    k_bounds: Tuple[int, int],
    noise_max: float = 0.50,
    max_class_share_max: float = 0.70,
    *,
    null_mode: str = "diagnostic",
    null_alpha: float = 0.05,
    reject_tautological: bool = True,
    diagnostic_backends: Sequence[str] = _DIAGNOSTIC_BACKENDS,
) -> Optional[str]:
    """``None`` si el método es viable; si no, el motivo del descarte.

    Los cinco primeros parámetros conservan nombre y posición: ``9_selection_audit``
    los pasa posicionalmente y lee el umbral del payload. Todo lo de v4 es
    keyword-only, de modo que una llamada vieja sigue funcionando.
    """
    if not metric.available:
        return "no produjo particion"
    if metric.diagnostic_only or str(metric.backend) in diagnostic_backends:
        backend = str(metric.backend or "?")
        return (
            f"backend de diagnostico {backend} (no seleccionable: no controla K"
            " y su ARI alto es artefacto del colapso)"
        )
    if not math.isfinite(float(metric.stability)):
        return "estabilidad no finita"
    lo, hi = int(k_bounds[0]), int(k_bounds[1])
    if not (lo <= int(metric.k) <= hi):
        return f"K={int(metric.k)} fuera de [{lo}, {hi}]"
    noise = float(metric.noise_frac)
    if noise > float(noise_max):
        return (
            f"ruido {noise:.3f} > max {float(noise_max):.2f} "
            "(mas de la mitad del corpus sin clasificar)"
        )
    share = float(metric.max_class_share or 0.0)
    dom = float(metric.dominance_corpus_effective)
    if dom > float(max_class_share_max):
        return (
            f"clase dominante {share:.2f} de las asignadas = {dom:.2f} del "
            f"corpus > max {float(max_class_share_max):.2f} (particion "
            f"degenerada: una clase y satelites; ruido {noise:.2f} descontado)"
        )
    usable = float(metric.usable_effective)
    if usable < float(coverage_floor):
        detail = f"coverage util {usable:.3f} < umbral {coverage_floor:.2f}"
        stable = float(metric.coverage_stable)
        if math.isfinite(stable):
            detail += f" (coverage estable {stable:.3f}"
        else:
            detail += f" (coverage {float(metric.coverage):.3f}"
        tiny = float(metric.tiny_frac or 0.0)
        if tiny > 0.0:
            detail += f" menos tiny {tiny:.3f}"
        return detail + ")"
    if (
        reject_tautological
        and metric.stability_kind == "perturbation"
        and str(metric.backend) in _TAUTOLOGICAL_BACKENDS
    ):
        return (
            f"estabilidad tautologica: backend {metric.backend} no tiene grados"
            " de libertad en el pase tight/loose (ARI=1.0 por construccion);"
            " falta bootstrap_ari"
        )
    boot = float(getattr(metric, "bootstrap_ari", float("nan")))
    if math.isfinite(boot) and boot < float(STABILITY_MEAN_MIN):
        return (
            f"sin senal de estructura: estabilidad media {boot:.3f} < piso "
            f"{float(STABILITY_MEAN_MIN):.2f} (el remuestreo no recupera la"
            " particion)"
        )
    if null_mode == "gate":
        p = float(metric.null_p)
        if math.isfinite(p) and p > float(null_alpha):
            return (
                f"sin contraste contra el nulo de DATOS: p={p:.3f} > alpha="
                f"{float(null_alpha):.2f} (las clases no separan mas que"
                " etiquetas permutadas)"
            )
    return None


def select_method(
    metrics: Sequence[MethodMetrics],
    margin: float = 0.02,
    coverage_floor: float = 0.15,
    k_bounds: Tuple[int, int] = (3, 20),
    min_class_share: float = 0.01,
    noise_max: float = 0.50,
    max_class_share_max: float = 0.70,
    null_mode: str = "diagnostic",
    null_alpha: float = 0.05,
    reject_tautological: bool = True,
    diagnostic_backends: Sequence[str] = _DIAGNOSTIC_BACKENDS,
) -> Selection:
    """Aplica la regla ``select-v4`` y devuelve la ``Selection``.

    Determinista: con los mismos ``metrics`` devuelve siempre lo mismo, incluso
    ante estabilidad y coverage idénticas (desempate por orden A<B<C). No lanza
    excepción con métricas incompletas: todo campo nuevo cae a ``nan`` = "no
    disponible" y su señal no interviene.
    """
    if null_mode not in _NULL_MODES:
        raise ValueError(
            f"null_mode={null_mode!r} no valido; usar uno de {_NULL_MODES}"
        )
    rows = list(metrics)
    common: Dict[str, Any] = dict(
        margin=float(margin),
        coverage_floor=float(coverage_floor),
        min_class_share=float(min_class_share),
        noise_max=float(noise_max),
        max_class_share_max=float(max_class_share_max),
        k_bounds=tuple(k_bounds),
        null_mode=str(null_mode),
        null_alpha=float(null_alpha),
    )
    if not rows:
        return Selection(
            winner=None,
            winner_ari=float("nan"),
            winner_coverage=float("nan"),
            winner_k=0,
            abstained=True,
            reason="no se recibieron metricas",
            **common,
        )

    discarded: List[Dict[str, Any]] = []
    viable: List[MethodMetrics] = []
    for m in rows:
        why = _feasibility(
            m,
            coverage_floor,
            k_bounds,
            noise_max,
            max_class_share_max,
            null_mode=null_mode,
            null_alpha=null_alpha,
            reject_tautological=reject_tautological,
            diagnostic_backends=diagnostic_backends,
        )
        if why is None:
            viable.append(m)
        else:
            discarded.append({"method": m.method, "why": why})

    all_rows = [m.as_row() for m in rows]

    if not viable:
        return Selection(
            winner=None,
            winner_ari=float("nan"),
            winner_coverage=float("nan"),
            winner_k=0,
            abstained=True,
            reason="ningun metodo cumple viabilidad",
            discarded=discarded,
            all_metrics=all_rows,
            **common,
        )

    def _rank_key(m: MethodMetrics) -> Tuple[Any, ...]:
        head: Tuple[Any, ...] = ()
        if null_mode == "rank":
            head = (_null_class(m, null_alpha),)
        return head + (
            -float(m.stability),
            -float(m.usable_effective),
            _METHOD_ORDER.get(m.method, 99),
        )

    ranked = sorted(viable, key=_rank_key)
    win = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None

    # ``delta_ari`` conserva el nombre por compatibilidad; con bootstrap_ari
    # disponible mide la diferencia de estabilidad, no la de ARI.
    delta_ari = float(win.stability - runner.stability) if runner else float("nan")
    delta_cov = (
        float(win.usable_effective - runner.usable_effective)
        if runner
        else float("nan")
    )
    ambiguous = bool(runner is not None and abs(delta_ari) <= float(margin))
    kinds_differ = bool(
        runner is not None and win.stability_kind != runner.stability_kind
    )
    coverage_conflict = bool(
        runner is not None
        and runner.usable_effective > win.usable_effective + 1e-12
    )
    conflict = bool(runner is not None and (kinds_differ or coverage_conflict))

    if runner is None:
        reason = "unico metodo viable"
    elif abs(delta_ari) <= 1e-12:
        if abs(delta_cov) > 1e-12:
            reason = "empate de estabilidad; desempate por coverage util"
        else:
            reason = "empate de estabilidad; desempate por orden canonico"
    else:
        reason = f"max estabilidad intra-metodo ({win.stability_kind})"
    if null_mode == "rank" and runner is not None:
        if _null_class(win, null_alpha) != _null_class(runner, null_alpha):
            reason = (
                "evidencia contra el nulo (0=si, 1=sin medir, 2=no); despues "
                f"estabilidad intra-metodo ({win.stability_kind})"
            )

    return Selection(
        winner=win.method,
        winner_ari=float(win.ari),
        winner_coverage=float(win.coverage),
        winner_k=int(win.k),
        # ``usable`` del ganador = cobertura efectiva (coincide con v3 cuando
        # ``coverage_stable`` no existe).
        winner_usable_coverage=float(win.usable_effective),
        runner_up=runner.method if runner else None,
        runner_up_ari=float(runner.ari) if runner else float("nan"),
        runner_up_coverage=float(runner.coverage) if runner else float("nan"),
        runner_up_k=int(runner.k) if runner else 0,
        runner_up_usable_coverage=(
            float(runner.usable_effective) if runner else float("nan")
        ),
        delta_ari=delta_ari,
        delta_coverage=delta_cov,
        ambiguous=ambiguous,
        criteria_conflict=conflict,
        abstained=False,
        reason=reason,
        discarded=discarded,
        all_metrics=all_rows,
        winner_stability=float(win.stability),
        winner_stability_kind=win.stability_kind,
        winner_null_p=float(win.null_p),
        winner_dominance_corpus=float(win.dominance_corpus_effective),
        runner_up_stability=float(runner.stability) if runner else float("nan"),
        runner_up_stability_kind=runner.stability_kind if runner else "",
        **common,
    )


# ══════════════════════════════════════════════════════════════════════════
# Persistencia
# ══════════════════════════════════════════════════════════════════════════
def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_sel_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def persist_selection(
    selection: Selection,
    path: str,
    history_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Escribe el snapshot (sobrescribe) y, si se pide, una línea de historial.

    El historial es JSONL *append-only*: una línea por corrida. Es la materia
    prima del auditor (``9_selection_audit.py``), que detecta si el ganador
    cambia entre especificaciones vecinas.
    """
    ctx = dict(context or {})
    ctx.setdefault("written_at", datetime.now(timezone.utc).isoformat())
    payload = {
        "rule_version": selection.rule_version,
        "context": ctx,
        "selection": selection.to_dict(),
    }
    _atomic_write_json(path, payload)
    if history_path:
        directory = os.path.dirname(os.path.abspath(history_path))
        os.makedirs(directory, exist_ok=True)
        record = dict(payload)
        record["context"] = ctx
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    return payload


#: Motivo del último snapshot ignorado por ``load_selection`` (o ``None``).
_LAST_LOAD_REJECTION: Optional[Dict[str, Any]] = None


def last_load_rejection() -> Optional[Dict[str, Any]]:
    """Motivo por el que ``load_selection`` ignoró el último snapshot."""
    return dict(_LAST_LOAD_REJECTION) if _LAST_LOAD_REJECTION else None


def load_selection(
    path: str,
    corpus_fingerprint: Optional[str] = None,
    uce_target_size: Optional[int] = None,
    expected_rule_version: Optional[str] = None,
    explain: Optional[List[str]] = None,
) -> Optional[Selection]:
    """Carga el snapshot. Devuelve ``None`` si no corresponde (invalidación).

    Tres invalidaciones explícitas, en este orden:

    1. ``rule_version`` distinta de ``expected_rule_version`` (por defecto la
       del módulo). Reutilizar un snapshot ``select-v3`` bajo ``select-v4``
       aplicaría una decisión tomada con otra regla **y parecería válida**,
       porque el corpus y el target coinciden: es exactamente el fallo que costó
       tiempo. Con ``expected_rule_version=""`` se desactiva el chequeo (solo
       para leer historia).
    2. ``corpus_fingerprint`` distinto.
    3. ``uce_target_size`` distinto.

    Cada rechazo se registra en ``last_load_rejection()`` y en el log del
    módulo; si se pasa ``explain``, se añade una línea legible.
    """
    global _LAST_LOAD_REJECTION
    _LAST_LOAD_REJECTION = None

    def _reject(why: str, declared: Any = None, expected: Any = None) -> None:
        global _LAST_LOAD_REJECTION
        _LAST_LOAD_REJECTION = {
            "path": str(path),
            "why": why,
            "declared": "" if declared is None else str(declared),
            "expected": "" if expected is None else str(expected),
        }
        logging.getLogger(__name__).info("snapshot ignorado: %s", why)
        if explain is not None:
            explain.append(f"[!] snapshot ignorado ({path}): {why}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        _reject(f"no se pudo leer ({type(exc).__name__})")
        return None

    sel = payload.get("selection")
    if not isinstance(sel, dict):
        _reject("payload sin 'selection'")
        return None

    expected = (
        RULE_VERSION if expected_rule_version is None else str(expected_rule_version)
    )
    if expected:
        declared = payload.get("rule_version", sel.get("rule_version"))
        if declared != expected:
            _reject(
                f"rule_version {declared!r} != {expected!r} (snapshot de otra regla)",
                declared,
                expected,
            )
            return None

    ctx = payload.get("context") or {}
    if corpus_fingerprint is not None:
        stored = ctx.get("corpus_fingerprint")
        if stored is not None and stored != corpus_fingerprint:
            _reject(f"corpus_fingerprint {stored!r} != {corpus_fingerprint!r}")
            return None
    if uce_target_size is not None:
        stored = ctx.get("uce_target_size")
        try:
            mismatch = int(stored) != int(uce_target_size)
        except (TypeError, ValueError):
            mismatch = True
        if mismatch:
            _reject(f"uce_target_size {stored!r} != {uce_target_size!r}")
            return None
    return Selection.from_dict(sel)


# ══════════════════════════════════════════════════════════════════════════
# Auditoría del historial
# ══════════════════════════════════════════════════════════════════════════
def read_history(path: str) -> List[Dict[str, Any]]:
    """Lee el JSONL del historial. Ignora líneas corruptas (nunca aborta)."""
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return records


def _fingerprint_of(record: Dict[str, Any]) -> str:
    ctx = record.get("context") or {}
    return str(ctx.get("corpus_fingerprint") or ctx.get("spec") or "?")


def _sort_key(record: Dict[str, Any]) -> Tuple[int, str]:
    ctx = record.get("context") or {}
    try:
        target = int(ctx.get("uce_target_size") or 0)
    except (TypeError, ValueError):
        target = 0
    return target, str(ctx.get("written_at") or "")


def audit_history(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Agrupa por corpus y detecta cambios de ganador entre targets vecinos.

    Devuelve ``{"groups": [...], "instability": [...], "ambiguous": [...]}``.
    Un grupo inestable significa que la elección **no es identificable** con
    esa segmentación: el ganador depende de cuántas UCEs haya, no de los datos.
    """
    by_corpus: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_corpus.setdefault(_fingerprint_of(rec), []).append(rec)

    groups: List[Dict[str, Any]] = []
    instability: List[Dict[str, Any]] = []
    ambiguous_rows: List[Dict[str, Any]] = []
    abstained_rows: List[Dict[str, Any]] = []

    for fp, recs in sorted(by_corpus.items()):
        ordered = sorted(recs, key=_sort_key)
        rows = []
        for rec in ordered:
            ctx = rec.get("context") or {}
            sel = rec.get("selection") or {}
            rows.append(
                {
                    "written_at": ctx.get("written_at"),
                    "target": ctx.get("uce_target_size"),
                    "mode": ctx.get("mode"),
                    "winner": sel.get("winner"),
                    "ari": sel.get("winner_ari"),
                    "coverage": sel.get("winner_coverage"),
                    "k": sel.get("winner_k"),
                    "ambiguous": bool(sel.get("ambiguous")),
                    "criteria_conflict": bool(sel.get("criteria_conflict")),
                    "abstained": bool(sel.get("abstained")) or not sel.get("winner"),
                }
            )
            if sel.get("ambiguous") or sel.get("criteria_conflict"):
                ambiguous_rows.append({"corpus": fp, **rows[-1]})
            if rows[-1]["abstained"]:
                abstained_rows.append({"corpus": fp, **rows[-1]})
        winners = [r["winner"] for r in rows if r["winner"]]
        distinct = sorted(set(winners))
        groups.append({"corpus": fp, "rows": rows, "winners": distinct})
        if len(distinct) > 1:
            instability.append(
                {
                    "corpus": fp,
                    "winners": distinct,
                    "by_target": {str(r["target"]): r["winner"] for r in rows},
                }
            )
    return {
        "groups": groups,
        "instability": instability,
        "ambiguous": ambiguous_rows,
        "abstained": abstained_rows,
        "n_records": sum(len(g["rows"]) for g in groups),
    }


def audit_text(records: Iterable[Dict[str, Any]]) -> str:
    """Render ASCII del auditor (sin cajas: consola cp1252)."""
    report = audit_history(records)
    lines = [
        f"SELECCION PERSISTIDA - {report['n_records']} corridas, "
        f"{len(report['groups'])} corpus"
    ]
    for group in report["groups"]:
        lines.append("")
        lines.append(
            f"corpus {group['corpus']}  (ganadores: {', '.join(group['winners']) or '-'})"
        )
        lines.append(
            f"  {'target':>6}  {'modo':10}  {'met':4}{'K':>4}{'ARI':>8}{'cov':>8}  flags"
        )
        for row in group["rows"]:
            flags = []
            if row["ambiguous"]:
                flags.append("AMBIGUO")
            if row["criteria_conflict"]:
                flags.append("conflicto")
            if row.get("abstained"):
                flags.append("SIN CLASIFICACION")
            ari = row["ari"]
            cov = row["coverage"]
            lines.append(
                f"  {str(row['target']):>6}  {str(row['mode'] or '-'):10}  "
                f"{str(row['winner'] or '-'):4}{int(row['k'] or 0):>4}"
                f"{(float(ari) if ari is not None else float('nan')):>8.3f}"
                f"{(float(cov) if cov is not None else float('nan')):>8.3f}  "
                f"{', '.join(flags) or '-'}"
            )
    if report["instability"]:
        lines.append("")
        lines.append("[!] el ganador CAMBIA dentro del mismo corpus:")
        for item in report["instability"]:
            pairs = ", ".join(f"{k}->{v}" for k, v in item["by_target"].items())
            lines.append(f"    corpus {item['corpus']}: {pairs}")
        lines.append(
            "    => la eleccion no es identificable con esta segmentacion; "
            "no persiste entre targets."
        )
    else:
        lines.append("")
        lines.append("[ok] el ganador es el mismo en todos los targets del corpus.")
    if report["abstained"]:
        lines.append("")
        lines.append(
            f"[!] {len(report['abstained'])} corrida(s) SIN CLASIFICACION "
            "(ningun metodo viable):"
        )
        for row in report["abstained"]:
            lines.append(
                f"    corpus {row['corpus']} target={row['target']} modo={row['mode']}"
            )
    if report["ambiguous"]:
        lines.append("")
        lines.append(
            f"[!] {len(report['ambiguous'])} corrida(s) con decision no holgada:"
        )
        for row in report["ambiguous"]:
            lines.append(
                f"    corpus {row['corpus']} target={row['target']}: "
                f"{row['winner']} (ARI={row['ari']})"
            )
    return "\n".join(lines)
