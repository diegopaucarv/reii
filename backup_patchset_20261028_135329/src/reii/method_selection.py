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

Regla (lexicográfica, versión ``select-v3``)
────────────────────────────────────────────
1. **Viabilidad**: el método existe (produjo partición), su ``K`` cae en
   ``k_bounds`` (por defecto **3..20**: ``K=2`` es trivial y su ARI=1.0 no
   significa nada), su **ruido** no supera ``noise_max`` (0.50: más de medio
   corpus sin clasificar no es una partición) y su **coverage usable** alcanza
   ``coverage_floor`` (por defecto **0.15**).
2. **Maximizar ARI intra-método** (A1↔A2, B1↔B2, C1↔C2). Es la única señal de
   reproducibilidad que no depende de comparar particiones distintas.
3. **Desempate por coverage usable** (más material analizable gana).
4. **Ambigüedad declarada**: si ``|ARI_1 - ARI_2| <= margin`` la elección se
   marca ``ambiguous``. Si además el segundo tiene **más coverage usable** que
   el ganador, se marca ``criteria_conflict`` (los dos criterios apuntan a
   métodos distintos). En ambos casos el ganador se mantiene —la regla es
   determinista— pero el run queda marcado para revisión humana.
5. **Abstención**: si ningún método es viable, ``winner=None`` y
   ``abstained=True``. El pipeline **no clasifica** pero continúa: las UCEs
   quedan sin ``cluster_id`` y las etapas posteriores corren sobre vacío.
6. **No dominancia** (select-v3): si la clase mayor se come más del
   ``max_class_share_max`` (90 %) de las UCEs **asignadas**, la partición es
   degenerada (una clase + satélites) y el método se descarta. El ARI de una
   partición así es alto por construcción: replicar una clase gigante es
   fácil. Se mide sobre lo asignado, no sobre el corpus, para que el ruido de
   HDBSCAN no maquille la dominancia.

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
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RULE_VERSION = "select-v3"

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

    @property
    def usable_coverage(self) -> float:
        """UCEs estables y en una clase con masa suficiente / ``n_total``."""
        return max(0.0, float(self.coverage) - float(self.tiny_frac))

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["usable"] = self.usable_coverage
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
    max_class_share_max: float = 0.90
    k_bounds: Tuple[int, int] = (3, 20)

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
        ):
            if d.get(key) is None:
                d[key] = float("nan")
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})

    # ── presentación (ASCII puro: la consola del host es cp1252) ─────────
    def explain_lines(self) -> List[str]:
        out = [
            f"METHOD SELECTION (rule {self.rule_version})",
            f"  {'met':4}{'K':>4}{'ARI':>8}{'cov':>7}{'usable':>8}{'noise':>7}"
            f"{'tiny':>7}{'dom':>7}  {'perturbacion':14}  veredicto",
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
                f"{float(row.get('ari') or 0.0):>8.3f}"
                f"{float(row.get('coverage') or 0.0):>7.3f}"
                f"{float(row.get('usable') or 0.0):>8.3f}"
                f"{float(row.get('noise_frac') or 0.0):>7.2f}"
                f"{float(row.get('tiny_frac') or 0.0):>7.2f}  "
                f"{float(row.get('max_class_share') or 0.0):>7.2f}  "
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
            f"     ARI={self.winner_ari:.3f}  coverage={self.winner_coverage:.3f}  "
            f"usable={self.winner_usable_coverage:.3f}  K={self.winner_k}"
        )
        _win_row = next((r for r in self.all_metrics if r["method"] == self.winner), {})
        out.append(
            f"     clase mayor={float(_win_row.get('max_class_share') or 0.0):.2f} "
            f"de las asignadas (gate: >{self.max_class_share_max:.2f} = "
            "particion degenerada)"
        )
        if self.ambiguous and self.runner_up:
            out.append(
                f"  [!] AMBIGUO: {self.winner} vs {self.runner_up}, "
                f"|dARI|={abs(self.delta_ari):.3f} <= margin={self.margin:.3f}"
            )
        if self.criteria_conflict and self.runner_up:
            out.append(
                f"  [!] conflicto de criterios: {self.runner_up} tiene mas coverage "
                f"usable ({self.runner_up_usable_coverage:.3f} vs "
                f"{self.winner_usable_coverage:.3f}); el ARI manda pero la eleccion "
                "no es holgada"
            )
        return out

    def explain_text(self) -> str:
        return "\n".join(self.explain_lines())


# ══════════════════════════════════════════════════════════════════════════
# Regla
# ══════════════════════════════════════════════════════════════════════════
def _feasibility(
    metric: MethodMetrics,
    coverage_floor: float,
    k_bounds: Tuple[int, int],
    noise_max: float = 0.50,
    max_class_share_max: float = 0.90,
) -> Optional[str]:
    """``None`` si el método es viable; si no, el motivo del descarte."""
    if not metric.available:
        return "no produjo particion"
    ari = float(metric.ari)
    if not math.isfinite(ari):
        return "ARI no finito"
    lo, hi = int(k_bounds[0]), int(k_bounds[1])
    if not (lo <= int(metric.k) <= hi):
        return f"K={int(metric.k)} fuera de [{lo}, {hi}]"
    if float(metric.noise_frac) > float(noise_max):
        return (
            f"ruido {float(metric.noise_frac):.3f} > max {float(noise_max):.2f} "
            "(mas de la mitad del corpus sin clasificar)"
        )
    dom = float(getattr(metric, "max_class_share", 0.0) or 0.0)
    if dom > float(max_class_share_max):
        return (
            f"clase dominante {dom:.2f} de las asignadas > max "
            f"{float(max_class_share_max):.2f} (particion degenerada: una clase"
            " y satelites)"
        )
    usable = metric.usable_coverage
    if usable < float(coverage_floor):
        detail = f"coverage util {usable:.3f} < umbral {coverage_floor:.2f}"
        tiny = float(getattr(metric, "tiny_frac", 0.0) or 0.0)
        if tiny > 0.0:
            detail += f" (coverage {float(metric.coverage):.3f} menos tiny {tiny:.3f})"
        return detail
    return None


def select_method(
    metrics: Sequence[MethodMetrics],
    margin: float = 0.02,
    coverage_floor: float = 0.15,
    k_bounds: Tuple[int, int] = (3, 20),
    min_class_share: float = 0.01,
    noise_max: float = 0.50,
    max_class_share_max: float = 0.90,
) -> Selection:
    """Aplica la regla ``select-v3`` y devuelve la ``Selection``.

    Determinista: con los mismos ``metrics`` devuelve siempre lo mismo,
    incluso ante ARI y coverage idénticos (desempate por orden A<B<C).
    """
    rows = list(metrics)
    if not rows:
        return Selection(
            winner=None,
            winner_ari=float("nan"),
            winner_coverage=float("nan"),
            winner_k=0,
            abstained=True,
            reason="no se recibieron metricas",
            margin=margin,
            coverage_floor=coverage_floor,
            min_class_share=min_class_share,
            noise_max=noise_max,
            max_class_share_max=max_class_share_max,
            k_bounds=tuple(k_bounds),
        )

    discarded: List[Dict[str, Any]] = []
    viable: List[MethodMetrics] = []
    for m in rows:
        why = _feasibility(m, coverage_floor, k_bounds, noise_max, max_class_share_max)
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
            margin=margin,
            coverage_floor=coverage_floor,
            min_class_share=min_class_share,
            noise_max=noise_max,
            max_class_share_max=max_class_share_max,
            k_bounds=tuple(k_bounds),
        )

    ranked = sorted(
        viable,
        key=lambda m: (
            -float(m.ari),
            -m.usable_coverage,
            _METHOD_ORDER.get(m.method, 99),
        ),
    )
    win = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None

    delta_ari = float(win.ari - runner.ari) if runner else float("nan")
    delta_cov = (
        float(win.usable_coverage - runner.usable_coverage) if runner else float("nan")
    )
    ambiguous = bool(runner is not None and abs(delta_ari) <= float(margin))
    conflict = bool(
        runner is not None and runner.usable_coverage > win.usable_coverage + 1e-12
    )

    reason = "max ARI intra-metodo"
    if runner is not None and abs(delta_ari) <= 1e-12:
        reason = "empate de ARI; desempate por coverage util"

    return Selection(
        winner=win.method,
        winner_ari=float(win.ari),
        winner_coverage=float(win.coverage),
        winner_k=int(win.k),
        winner_usable_coverage=float(win.usable_coverage),
        runner_up=runner.method if runner else None,
        runner_up_ari=float(runner.ari) if runner else float("nan"),
        runner_up_coverage=float(runner.coverage) if runner else float("nan"),
        runner_up_k=int(runner.k) if runner else 0,
        runner_up_usable_coverage=(
            float(runner.usable_coverage) if runner else float("nan")
        ),
        delta_ari=delta_ari,
        delta_coverage=delta_cov,
        ambiguous=ambiguous,
        criteria_conflict=conflict,
        abstained=False,
        reason=reason,
        discarded=discarded,
        all_metrics=all_rows,
        margin=float(margin),
        coverage_floor=float(coverage_floor),
        min_class_share=float(min_class_share),
        noise_max=float(noise_max),
        max_class_share_max=float(max_class_share_max),
        k_bounds=tuple(k_bounds),
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


def load_selection(
    path: str,
    corpus_fingerprint: Optional[str] = None,
    uce_target_size: Optional[int] = None,
) -> Optional[Selection]:
    """Carga el snapshot. Devuelve ``None`` si falta o si no corresponde al
    corpus/segmentación pedidos (invalidación explícita)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    ctx = payload.get("context") or {}
    if corpus_fingerprint is not None:
        stored = ctx.get("corpus_fingerprint")
        if stored is not None and stored != corpus_fingerprint:
            return None
    if uce_target_size is not None:
        stored = ctx.get("uce_target_size")
        if stored is not None and int(stored) != int(uce_target_size):
            return None
    sel = payload.get("selection")
    if not isinstance(sel, dict):
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
