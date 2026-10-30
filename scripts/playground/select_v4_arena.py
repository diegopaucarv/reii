#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tablero de escenarios de la regla de seleccion ``select-v4``.

QUE ES
------
Una tabla de 16 escenarios con metricas *fijas* (sin clustering, sin numpy, sin
red) que muestra, de un vistazo, **quien gana y por que**. Sirve para revisar la
regla sin correr el pipeline ni esperar a Stanza: cada fila aísla UNA decision.

Las metricas de cada escenario se eligen para que el veredicto dependa de una
sola senal nueva (estabilidad, nulo, dominancia, ruido, backend). No son
sinteticas al azar: los numeros de ``dominancia`` (0.837), ``noise`` y los ARI
de A/B provienen de ``output.log`` y del probe v4
(``docs/design_v5/GATE_NOISE.md`` seccion 10).

QUE IMPORTA Y QUE NO
--------------------
* NO importa ``reii.main_workflow`` (Stanza) ni ``reii.quality``.
* ``reii.method_selection`` se importa SOLO para la columna ``v3``, y es
  liviano (stdlib puro). Si el modulo ya es ``select-v4``, la columna v3 se
  omite (no se puede reconstruir v3 con el codigo nuevo sin mentir).
* No escribe en ``data/``, ``output/``, ``models/`` ni ``src/``.
* Salida ASCII puro (consola cp1252), prefijos ``->``, ``[ok]``, ``[!]``, ``[X]``,
  ``=>``. Exit 0 si todos los escenarios coinciden con lo esperado.

COPIA CANONICA
--------------
La seccion "COPIA CANONICA" es el **mismo texto** que el diff de
``docs/design_v5/SELECT_V4.md`` (``MethodMetrics`` + ``_feasibility`` +
``select_method``). Vive aqui porque las reglas de trabajo prohiben escribir en
``src/**``; cuando se aplique el diff, ``--use-module`` compara la copia con el
modulo real y avisa si divergen. ``Selection`` se sustituye por ``Sel``, un
espejo con los mismos nombres de campo que solo anade el render de la tabla.

Uso:
    python scripts/playground/select_v4_arena.py
    python scripts/playground/select_v4_arena.py --use-module   # tras el diff
    python scripts/playground/select_v4_arena.py --verbose
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

NAN = float("nan")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")


# ══════════════════════════════════════════════════════════════════════════
# COPIA CANONICA de la regla select-v4 (= diff de SELECT_V4.md, seccion 7)
# ══════════════════════════════════════════════════════════════════════════
RULE_VERSION = "select-v4"

#: Orden canonico de desempate final (determinista).
_METHOD_ORDER = {"A": 0, "B": 1, "C": 2}

#: Backends sin grados de libertad en el pase tight/loose: ``ari == 1.0`` por
#: construccion (BACKEND_C.md seccion 3), asi que ese ARI no mide estabilidad.
_TAUTOLOGICAL_BACKENDS = ("ward", "kmeans_spherical")

#: Modos de uso del contraste contra el nulo.
_NULL_MODES = ("gate", "rank", "off")


@dataclass(frozen=True)
class MM:
    """Espejo de ``MethodMetrics`` post-diff (mismos campos y propiedades)."""

    method: str
    k: int
    ari: float
    coverage: float
    noise_frac: float = 0.0
    tiny_frac: float = 0.0
    perturbation: str = ""
    available: bool = True
    assigned_frac: Optional[float] = None
    min_class_size: int = 0
    median_class_size: float = 0.0
    max_class_share: float = 0.0
    # ── select-v4: senales nuevas (defaults = "no disponible") ───────────
    backend: str = ""
    diagnostic_only: bool = False
    bootstrap_ari: float = NAN
    bootstrap_ci_lo: float = NAN
    bootstrap_ci_hi: float = NAN
    coverage_stable: float = NAN
    null_p: float = NAN
    null_z: float = NAN
    n_total: int = 0
    n_assigned: int = 0
    n_noise: int = 0
    max_class_size: int = 0
    dominance_corpus: float = NAN

    @property
    def usable_coverage(self) -> float:
        """UCEs estables y en una clase con masa suficiente / ``n_total``."""
        return max(0.0, float(self.coverage) - float(self.tiny_frac))

    @property
    def coverage_effective(self) -> float:
        """``coverage_stable`` si el motor la produjo; si no ``coverage``."""
        stable = float(self.coverage_stable)
        return stable if math.isfinite(stable) else float(self.coverage)

    @property
    def usable_effective(self) -> float:
        """Cobertura efectiva descontando clases diminutas."""
        return max(0.0, self.coverage_effective - float(self.tiny_frac))

    @property
    def dominance_corpus_effective(self) -> float:
        """Clase mayor / ``n_total``; se deriva si el productor no la manda."""
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
        """Criterio principal: bootstrap si existe, perturbacion si no."""
        boot = float(self.bootstrap_ari)
        return boot if math.isfinite(boot) else float(self.ari)

    @property
    def stability_kind(self) -> str:
        return (
            "bootstrap" if math.isfinite(float(self.bootstrap_ari)) else "perturbation"
        )


def _null_class(metric: MM, null_alpha: float) -> int:
    """0 = evidencia contra el nulo, 1 = sin medir, 2 = sin evidencia."""
    p = float(metric.null_p)
    if not math.isfinite(p):
        return 1
    return 0 if p <= float(null_alpha) else 2


def _feasibility(
    metric: MM,
    coverage_floor: float,
    k_bounds: Tuple[int, int],
    noise_max: float = 0.50,
    max_class_share_max: float = 0.70,
    *,
    null_mode: str = "gate",
    null_alpha: float = 0.05,
    reject_tautological: bool = True,
) -> Optional[str]:
    """``None`` si el metodo es viable; si no, el motivo del descarte."""
    if not metric.available:
        return "no produjo particion"
    if bool(metric.diagnostic_only):
        backend = str(metric.backend or "?")
        return (
            f"backend de diagnostico {backend} (no seleccionable: no controla K"
            " y su ARI alto es artefacto del colapso)"
        )
    stability = float(metric.stability)
    if not math.isfinite(stability):
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
        tiny = float(metric.tiny_frac or 0.0)
        if math.isfinite(float(metric.coverage_stable)):
            detail += f" (coverage estable {float(metric.coverage_stable):.3f}"
        else:
            detail += f" (coverage {float(metric.coverage):.3f}"
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
    if null_mode == "gate":
        p = float(metric.null_p)
        if math.isfinite(p) and p > float(null_alpha):
            return (
                f"sin contraste contra el nulo: p={p:.3f} > alpha="
                f"{float(null_alpha):.2f} (las clases no separan mas que"
                " etiquetas permutadas)"
            )
    return None


@dataclass
class Sel:
    """Espejo reducido de ``Selection`` (mismos nombres de campo)."""

    winner: Optional[str]
    winner_ari: float
    winner_coverage: float
    winner_k: int
    winner_usable_coverage: float = NAN
    runner_up: Optional[str] = None
    runner_up_ari: float = NAN
    runner_up_coverage: float = NAN
    runner_up_k: int = 0
    runner_up_usable_coverage: float = NAN
    delta_ari: float = NAN
    delta_coverage: float = NAN
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
    max_class_share_max: float = 0.70
    k_bounds: Tuple[int, int] = (3, 20)
    dominance_scale: str = "corpus"
    null_alpha: float = 0.05
    null_mode: str = "gate"
    winner_stability: float = NAN
    winner_stability_kind: str = ""
    winner_null_p: float = NAN
    winner_dominance_corpus: float = NAN
    runner_up_stability: float = NAN
    runner_up_stability_kind: str = ""


def select_method(
    metrics: Sequence[MM],
    margin: float = 0.02,
    coverage_floor: float = 0.15,
    k_bounds: Tuple[int, int] = (3, 20),
    min_class_share: float = 0.01,
    noise_max: float = 0.50,
    max_class_share_max: float = 0.70,
    null_mode: str = "gate",
    null_alpha: float = 0.05,
    reject_tautological: bool = True,
) -> Sel:
    """Aplica la regla ``select-v4`` y devuelve la ``Selection``."""
    if null_mode not in _NULL_MODES:
        raise ValueError(
            f"null_mode={null_mode!r} no valido; usar uno de {_NULL_MODES}"
        )
    rows = list(metrics)
    common = dict(
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
        return Sel(
            winner=None,
            winner_ari=NAN,
            winner_coverage=NAN,
            winner_k=0,
            abstained=True,
            reason="no se recibieron metricas",
            **common,
        )

    discarded: List[Dict[str, Any]] = []
    viable: List[MM] = []
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
        )
        if why is None:
            viable.append(m)
        else:
            discarded.append({"method": m.method, "why": why})

    all_rows = [
        {
            "method": m.method,
            "k": int(m.k),
            "ari": float(m.ari),
            "coverage": float(m.coverage),
            "coverage_stable": float(m.coverage_stable),
            "usable": float(m.usable_coverage),
            "usable_eff": float(m.usable_effective),
            "noise_frac": float(m.noise_frac),
            "tiny_frac": float(m.tiny_frac),
            "max_class_share": float(m.max_class_share),
            "dominance_corpus": float(m.dominance_corpus_effective),
            "stability": float(m.stability),
            "stability_kind": m.stability_kind,
            "bootstrap_ari": float(m.bootstrap_ari),
            "null_p": float(m.null_p),
            "null_z": float(m.null_z),
            "backend": str(m.backend),
            "diagnostic_only": bool(m.diagnostic_only),
            "perturbation": str(m.perturbation),
            "available": bool(m.available),
        }
        for m in rows
    ]

    if not viable:
        return Sel(
            winner=None,
            winner_ari=NAN,
            winner_coverage=NAN,
            winner_k=0,
            abstained=True,
            reason="ningun metodo cumple viabilidad",
            discarded=discarded,
            all_metrics=all_rows,
            **common,
        )

    if null_mode == "rank":
        ranked = sorted(
            viable,
            key=lambda m: (
                _null_class(m, null_alpha),
                -float(m.stability),
                -float(m.usable_effective),
                _METHOD_ORDER.get(m.method, 99),
            ),
        )
    else:
        ranked = sorted(
            viable,
            key=lambda m: (
                -float(m.stability),
                -float(m.usable_effective),
                _METHOD_ORDER.get(m.method, 99),
            ),
        )
    win = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None

    delta_ari = float(win.stability - runner.stability) if runner else NAN
    delta_cov = float(win.usable_effective - runner.usable_effective) if runner else NAN
    ambiguous = bool(runner is not None and abs(delta_ari) <= float(margin))
    kinds_differ = bool(
        runner is not None and win.stability_kind != runner.stability_kind
    )
    coverage_conflict = bool(
        runner is not None and runner.usable_effective > win.usable_effective + 1e-12
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

    return Sel(
        winner=win.method,
        winner_ari=float(win.ari),
        winner_coverage=float(win.coverage),
        winner_k=int(win.k),
        winner_usable_coverage=float(win.usable_coverage),
        runner_up=runner.method if runner else None,
        runner_up_ari=float(runner.ari) if runner else NAN,
        runner_up_coverage=float(runner.coverage) if runner else NAN,
        runner_up_k=int(runner.k) if runner else 0,
        runner_up_usable_coverage=(float(runner.usable_effective) if runner else NAN),
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
        runner_up_stability=float(runner.stability) if runner else NAN,
        runner_up_stability_kind=runner.stability_kind if runner else "",
        **common,
    )


# ══════════════════════════════════════════════════════════════════════════
# Escenarios
# ══════════════════════════════════════════════════════════════════════════
def _a(k: int, ari: float, cov: float, **kw: Any) -> MM:
    return MM("A", k, ari, cov, perturbation="A1<->A2", **kw)


def _b(k: int, ari: float, cov: float, **kw: Any) -> MM:
    return MM("B", k, ari, cov, perturbation="B1<->B2", **kw)


@dataclass
class Scenario:
    sid: str
    note: str
    metrics: List[MM]
    expect: Optional[str]
    kwargs: Dict[str, Any] = field(default_factory=dict)


def _scenarios() -> List[Scenario]:
    """16 escenarios, cada uno aislando una decision de la regla."""
    return [
        Scenario(
            "base_lex",
            "A gana por ARI; nada nuevo interviene (compatibilidad v3)",
            [
                _a(7, 0.350, 0.606),
                _b(9, 0.212, 0.415),
            ],
            "A",
        ),
        Scenario(
            "conflict_cov",
            "A gana por estabilidad pero B cubre mas: ambiguo + conflicto",
            [
                _a(6, 0.110, 0.267),
                _b(8, 0.073, 0.375),
            ],
            "A",
            {"margin": 0.05},
        ),
        Scenario(
            "c_hdbscan",
            "C con hdbscan nunca compite, aunque su ARI/coverage sean ideales",
            [
                _a(7, 0.300, 0.400),
                MM(
                    "C",
                    5,
                    0.990,
                    0.900,
                    noise_frac=0.12,
                    perturbation="C1<->C2",
                    backend="hdbscan",
                    diagnostic_only=True,
                    max_class_share=0.40,
                ),
            ],
            "A",
        ),
        Scenario(
            "c_ward_taut",
            "C ward sin bootstrap: ARI=1.0 tautologico, no es estabilidad",
            [
                _a(4, 0.450, 0.400),
                MM(
                    "C",
                    4,
                    1.000,
                    1.000,
                    perturbation="C1<->C2 (identico)",
                    backend="ward",
                    coverage_stable=0.95,
                    max_class_share=0.40,
                    null_p=0.004,
                ),
            ],
            "A",
        ),
        Scenario(
            "c_dense_dom",
            "C denso a=0.837, nu=0: c=0.837 > 0.70, descartado por dominancia",
            [
                _a(4, 0.250, 0.574),
                MM(
                    "C",
                    3,
                    0.785,
                    0.850,
                    perturbation="bootstrap(80%)",
                    backend="dense_cdh",
                    bootstrap_ari=0.620,
                    coverage_stable=0.800,
                    max_class_share=0.837,
                    null_p=0.004,
                ),
            ],
            "A",
        ),
        Scenario(
            "c_bad_null",
            "C con bootstrap decente pero p=0.61: sin evidencia contra el nulo",
            [
                _a(5, 0.400, 0.500, bootstrap_ari=0.400, null_p=0.010),
                MM(
                    "C",
                    4,
                    0.700,
                    1.000,
                    perturbation="bootstrap(80%)",
                    backend="dense_cdh",
                    bootstrap_ari=0.700,
                    coverage_stable=0.900,
                    max_class_share=0.400,
                    null_p=0.610,
                ),
            ],
            "A",
        ),
        Scenario(
            "c_dense_real",
            "C denso legitimo: bootstrap 0.72, evidencia fuerte; gana (no es anti-C)",
            [
                _a(4, 0.300, 0.400, bootstrap_ari=0.300, null_p=0.020),
                MM(
                    "C",
                    4,
                    0.960,
                    1.000,
                    perturbation="bootstrap(80%)",
                    backend="ward",
                    bootstrap_ari=0.720,
                    coverage_stable=0.900,
                    max_class_share=0.410,
                    null_p=0.004,
                ),
            ],
            "C",
        ),
        Scenario(
            "c_low_boot",
            "Mismo tipo (bootstrap): gana A por estabilidad, pero C cubre mas",
            [
                _a(5, 0.550, 0.600, bootstrap_ari=0.550, null_p=0.010),
                MM(
                    "C",
                    4,
                    0.250,
                    1.000,
                    perturbation="bootstrap(80%)",
                    backend="dense_cdh",
                    bootstrap_ari=0.250,
                    coverage_stable=0.850,
                    max_class_share=0.450,
                    null_p=0.010,
                ),
            ],
            "A",
        ),
        Scenario(
            "null_noise",
            "Ningun metodo supera el nulo: abstencion (no se clasifica)",
            [
                _a(5, 0.120, 0.300, null_p=0.620),
                _b(6, 0.100, 0.350, null_p=0.550),
                MM(
                    "C",
                    4,
                    0.200,
                    1.000,
                    perturbation="bootstrap(80%)",
                    backend="ward",
                    bootstrap_ari=0.200,
                    coverage_stable=0.500,
                    max_class_share=0.380,
                    null_p=0.400,
                ),
            ],
            None,
        ),
        Scenario(
            "k_floor",
            "C con K=2 (ARI=1.0 trivial) no es viable: K fuera de [3,20]",
            [
                _a(7, 0.300, 0.400),
                MM(
                    "C",
                    2,
                    1.000,
                    1.000,
                    backend="dense_cdh",
                    coverage_stable=0.950,
                    max_class_share=0.500,
                    null_p=0.004,
                ),
            ],
            "A",
        ),
        Scenario(
            "noise",
            "C denso con 60% sin asignar: mas de medio corpus no es particion",
            [
                _a(7, 0.300, 0.400),
                MM(
                    "C",
                    5,
                    0.990,
                    0.400,
                    noise_frac=0.600,
                    backend="dense_cdh",
                    bootstrap_ari=0.800,
                    coverage_stable=0.350,
                    max_class_share=0.500,
                    null_p=0.004,
                ),
            ],
            "A",
        ),
        Scenario(
            "cov_stable_low",
            "C estable en ARI pero solo 10% de UCEs estables: coverage efectiva",
            [
                _a(4, 0.300, 0.400),
                MM(
                    "C",
                    4,
                    0.700,
                    1.000,
                    perturbation="bootstrap(80%)",
                    backend="ward",
                    bootstrap_ari=0.700,
                    coverage_stable=0.100,
                    max_class_share=0.400,
                    null_p=0.004,
                ),
            ],
            "A",
        ),
        Scenario(
            "kinds_differ",
            "A mide perturbacion y C bootstrap: |d|=0.01 -> ambiguo y conflicto",
            [
                _a(6, 0.550, 0.500),
                MM(
                    "C",
                    4,
                    0.540,
                    1.000,
                    perturbation="bootstrap(80%)",
                    backend="dense_cdh",
                    bootstrap_ari=0.540,
                    coverage_stable=0.600,
                    max_class_share=0.300,
                    null_p=0.004,
                ),
            ],
            "A",
        ),
        Scenario(
            "only_taut",
            "Unico candidato es C ward sin bootstrap: abstencion, no victoria",
            [
                MM(
                    "C",
                    4,
                    1.000,
                    1.000,
                    backend="ward",
                    coverage_stable=0.950,
                    max_class_share=0.400,
                ),
            ],
            None,
        ),
        Scenario(
            "minimal",
            "Metricas minimas MethodMetrics('A',1,0.9,0.9): no lanzan excepcion",
            [MM("A", 1, 0.900, 0.900)],
            None,
        ),
        Scenario(
            "null_only_gate",
            "Unico candidato con p=0.61: el modo gate abstiene (default)",
            [
                _a(4, 0.400, 0.050, null_p=0.010),
                MM(
                    "C",
                    4,
                    0.700,
                    1.000,
                    perturbation="bootstrap(80%)",
                    backend="dense_cdh",
                    bootstrap_ari=0.700,
                    coverage_stable=0.900,
                    max_class_share=0.400,
                    null_p=0.610,
                ),
            ],
            None,
        ),
        Scenario(
            "null_only_rank",
            "Mismas metricas con null_mode='rank': C no se descarta y gana",
            [
                _a(4, 0.400, 0.050, null_p=0.010),
                MM(
                    "C",
                    4,
                    0.700,
                    1.000,
                    perturbation="bootstrap(80%)",
                    backend="dense_cdh",
                    bootstrap_ari=0.700,
                    coverage_stable=0.900,
                    max_class_share=0.400,
                    null_p=0.610,
                ),
            ],
            "C",
            {"null_mode": "rank"},
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# Render
# ══════════════════════════════════════════════════════════════════════════
def _v3_pick(metrics: Sequence[MM]) -> Optional[str]:
    """Selector v3 real, si el modulo instalado sigue siendo select-v3."""
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    try:
        from reii import method_selection as ms
    except Exception:
        return None
    if getattr(ms, "RULE_VERSION", "") != "select-v3":
        return None
    rows = [
        ms.MethodMetrics(
            method=m.method,
            k=int(m.k),
            ari=float(m.ari),
            coverage=float(m.coverage),
            noise_frac=float(m.noise_frac),
            tiny_frac=float(m.tiny_frac),
            perturbation=str(m.perturbation),
            available=bool(m.available),
            min_class_size=int(m.min_class_size),
            median_class_size=float(m.median_class_size),
            max_class_share=float(m.max_class_share),
        )
        for m in metrics
    ]
    sel = ms.select_method(rows, max_class_share_max=0.90)
    return sel.winner


def _flags(sel: Any) -> str:
    parts = []
    if sel.abstained:
        parts.append("abst")
    if sel.ambiguous:
        parts.append("amb")
    if sel.criteria_conflict:
        parts.append("conf")
    return "+".join(parts) or "-"


def _decision_note(sel: Any) -> str:
    if sel.winner is None:
        return sel.reason
    bits = [f"{sel.reason}"]
    if sel.runner_up:
        bits.append(
            f"d={sel.delta_ari:+.3f} ({sel.winner_stability:.3f} vs "
            f"{sel.runner_up_stability:.3f})"
        )
    else:
        bits.append(f"stab={sel.winner_stability:.3f} {sel.winner_stability_kind}")
    if sel.criteria_conflict:
        bits.append("kinds/coverage en conflicto")
    return "; ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-module", action="store_true", help="comparar con src/")
    ap.add_argument("--verbose", action="store_true", help="mostrar discards")
    args = ap.parse_args()

    scenarios = _scenarios()
    print(f"SELECT-V4 ARENA | {len(scenarios)} escenarios | regla {RULE_VERSION}")
    print("  A = word-count, B = similitud/coref, C = embeddings (denso)")
    print("")

    header = f"  {'escenario':16}{'v3':>4}{'v4':>6}{'run':>6}  {'flags':12}motivo"
    print(header)
    print("  " + "-" * (len(header) - 2))

    ok = 0
    rows_out: List[Tuple[str, Any]] = []
    for sc in scenarios:
        sel = select_method(sc.metrics, **sc.kwargs)
        v3 = _v3_pick(sc.metrics)
        got = sel.winner
        good = got == sc.expect
        ok += int(good)
        rows_out.append((sc.sid, sel))
        print(
            f"  {sc.sid:16}{str(v3 if v3 is not None else '-'):>4}"
            f"{str(got):>6}{str(sel.runner_up or '-'):>6}  "
            f"{_flags(sel):12}{_decision_note(sel)}"
        )
        if not good:
            print(f"  [!] {sc.sid}: esperado {sc.expect} y salio {got}")
            for d in sel.discarded:
                print(f"      {d['method']} descartado: {d['why']}")

    print("")
    print("  descartes (por que pierde cada metodo)")
    print("")
    for sc in scenarios:
        sel = dict(rows_out)[sc.sid]
        if not sel.discarded and not args.verbose:
            print(f"  {sc.sid:16}-> sin descartes (todos viables)")
            continue
        for d in sel.discarded:
            print(f"  {sc.sid:16}[!] {d['method']} descartado: {d['why']}")

    print("")
    print("  leyenda: stab = bootstrap_ari si existe, si no ari (2pass)")
    print("            domC = a*(1-nu) = clase mayor / n_total")
    print("            v3 = ganador con el modulo actual ('-' si ya es select-v4)")
    print("")

    if args.use_module:
        if SRC_DIR not in sys.path:
            sys.path.insert(0, SRC_DIR)
        try:
            from reii import method_selection as mod
        except Exception as exc:  # pragma: no cover
            print(f"  [!] no se pudo importar reii.method_selection: {exc}")
            return 1
        if getattr(mod, "RULE_VERSION", "") != RULE_VERSION:
            print(
                f"  [!] el modulo declara {getattr(mod, 'RULE_VERSION', '?')};"
                f" la copia canonica es {RULE_VERSION}: diff no aplicado"
            )
        else:
            diffs = 0
            for sc in scenarios:
                a = select_method(sc.metrics, **sc.kwargs)
                b = _module_select(mod, sc)
                if a.winner != b:
                    diffs += 1
                    print(f"  [X] {sc.sid}: copia={a.winner} modulo={b}")
            print(
                f"  [ok] copia canonica == modulo en {len(scenarios) - diffs}"
                f"/{len(scenarios)} escenarios"
                if not diffs
                else f"  [X] {diffs} divergencias entre copia y modulo"
            )
            if diffs:
                return 1

    total = len(scenarios)
    print(
        f"  {'[ok]' if ok == total else '[X]'} {ok}/{total} escenarios como se esperaba"
    )
    print("")
    return 0 if ok == total else 1


def _module_select(mod: Any, sc: Scenario) -> Optional[str]:
    """Ejecuta la regla del modulo real sobre un escenario (mismos campos)."""
    rows = [
        mod.MethodMetrics(
            method=m.method,
            k=int(m.k),
            ari=float(m.ari),
            coverage=float(m.coverage),
            noise_frac=float(m.noise_frac),
            tiny_frac=float(m.tiny_frac),
            perturbation=str(m.perturbation),
            available=bool(m.available),
            min_class_size=int(m.min_class_size),
            median_class_size=float(m.median_class_size),
            max_class_share=float(m.max_class_share),
            backend=str(m.backend),
            diagnostic_only=bool(m.diagnostic_only),
            bootstrap_ari=float(m.bootstrap_ari),
            coverage_stable=float(m.coverage_stable),
            null_p=float(m.null_p),
            null_z=float(m.null_z),
            n_total=int(m.n_total),
            n_assigned=int(m.n_assigned),
            n_noise=int(m.n_noise),
            max_class_size=int(m.max_class_size),
            dominance_corpus=float(m.dominance_corpus),
        )
        for m in sc.metrics
    ]
    return mod.select_method(rows, **sc.kwargs).winner


if __name__ == "__main__":
    sys.exit(main())
