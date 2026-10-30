#!/usr/bin/env python3
"""Medición offline de políticas de formas activas (fidelidad ALCESTE 2012).

No importa ``reii.main_workflow``, no corre Stanza, no usa red y es
determinista (semilla fija). Regla pre-registrada y veredicto:
``docs/design_v5/RETENTION_POLICY.md``.

Modos
-----
1. ``--token-table`` (exacto): JSONL escrito por ``TokenTableWriter`` con
   ``{forma, lemma, pos, is_stop, is_alpha}`` de TODOS los tokens. Requiere una
   pasada previa del pipeline (~13 s/UCE), pero mide sin aproximaciones.
2. Sin ``--token-table`` (aproximado): reconstruye desde ``workflow_data.json``

   - ``formas_tokens``                        -> activas del baseline
   - ``marcadores`` con POS de contenido      -> rescatadas al quitar ``is_stop``
   - ``marcadores`` con POS PROPN/X           -> rescatadas por ``pos_plus_propn``
   - residuo ``tokens - (formas U marcadores)`` menos numerales -> PROPN

   Esta reconstrucción reproduce las cifras publicadas (10.72 / 13.16 / 12.24
   sobre 528 UCEs); el script lo verifica e imprime ``VALIDADA`` o ``FALLA``.

Uso
---
    python scripts/retention_policy_report.py --input data/workflow_data.json
    python scripts/retention_policy_report.py --input data/workflow_data.json \
        --token-table data/token_table.jsonl --bootstrap 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from reii.retention_policy import (  # noqa: E402
    BASE_CONTENT_POS,
    PROPN_LIKE_POS,
    TOOL_ADVERB_LEMMAS,
    fold,
    is_tool_adverb,
    policy_content_pos,
    read_token_table,
    token_decision,
)

POLICIES = ("pos_only", "pos_plus_propn", "tool_adverb_filtered")
LABELS = {
    "pos_only": "pos_only (baseline)",
    "pos_plus_propn": "pos_plus_propn (agresiva)",
    "tool_adverb_filtered": "tool_adverb_filtered (conservadora)",
}

ALCESTE_LEMMAS_PER_UCE = 13.95
ALCESTE_RETENTION = 0.537
TOOL_QUOTA_MAX = 0.30
QUOTA_SENSITIVITY = (0.15, 0.25, 0.30, 0.35)

# Contadores publicados en docs/design_v5/RETENTION_ROOTCAUSE.md §2 (528 UCEs).
PUBLISHED_COUNTS = {
    "pos_only": 5658,
    "pos_plus_propn": 6949,
    "tool_adverb_filtered": 6465,
}
PUBLISHED_UCES = 528

# Numerales y fraccionarios: spaCy los etiqueta NUM y NO entran en ninguna
# política (tampoco en ALCESTE). En el residuo hay que descartarlos para no
# contarlos como PROPN. Lista cerrada, aplicada sobre formas plegadas.
NUMERAL_FORMS = frozenset(
    {
        "cero",
        "uno",
        "una",
        "un",
        "dos",
        "tres",
        "cuatro",
        "cinco",
        "seis",
        "siete",
        "ocho",
        "nueve",
        "diez",
        "once",
        "doce",
        "trece",
        "catorce",
        "quince",
        "dieciseis",
        "diecisiete",
        "dieciocho",
        "diecinueve",
        "veinte",
        "veintiuno",
        "veintiun",
        "veintidos",
        "veintitres",
        "veinticuatro",
        "veinticinco",
        "veintiseis",
        "veintisiete",
        "veintiocho",
        "veintinueve",
        "treinta",
        "cuarenta",
        "cincuenta",
        "sesenta",
        "setenta",
        "ochenta",
        "noventa",
        "cien",
        "ciento",
        "doscientos",
        "trescientos",
        "cuatrocientos",
        "quinientos",
        "seiscientos",
        "setecientos",
        "ochocientos",
        "novecientos",
        "mil",
        "millon",
        "millones",
        "billon",
        "billones",
        "medio",
        "media",
        "tercio",
        "cuarto",
        "mitad",
    }
)


# ──────────────────────────────────────────────────────────────────────
# Carga de datos
# ----------------------------------------------------------------------
def load_workflow(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    uces = data.get("uces") if isinstance(data, dict) else data
    if not isinstance(uces, list):
        raise SystemExit("La entrada no contiene una lista de UCEs: {}".format(path))
    return uces


# ──────────────────────────────────────────────────────────────────────
# Modo 2: reconstrucción por residuo
# ----------------------------------------------------------------------
def _residue_propn(uce: Dict[str, Any]) -> List[str]:
    """Formas del residuo que no son numerales (candidatas a PROPN/X)."""
    used: Counter = Counter()
    for forma in uce.get("formas_tokens") or []:
        used[fold(forma.get("forma"))] += 1
    for marcador in uce.get("marcadores") or []:
        used[fold(marcador.get("forma"))] += 1

    tokens: Counter = Counter(fold(token) for token in (uce.get("tokens") or []))
    residue = tokens - used

    out: List[str] = []
    for form, count in residue.items():
        if count <= 0 or not form or form in NUMERAL_FORMS:
            continue
        out.extend([form] * count)
    return out


def build_residue_records(uces: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for uce in uces:
        words = len((uce.get("texto") or "").split())
        if words == 0:
            words = len(uce.get("tokens") or [])

        base = [fold(item.get("forma")) for item in (uce.get("formas_tokens") or [])]
        base = [form for form in base if form]
        rescued: List[str] = []
        rescued_tool: List[str] = []
        propn: List[str] = []

        for marcador in uce.get("marcadores") or []:
            pos = marcador.get("pos")
            form = fold(marcador.get("forma"))
            if not form:
                continue
            lemma = marcador.get("lemma") or marcador.get("forma")
            if pos in BASE_CONTENT_POS:
                rescued.append(form)
                if is_tool_adverb("tool_adverb_filtered", pos, lemma):
                    rescued_tool.append(form)
            elif pos in PROPN_LIKE_POS:
                propn.append(form)

        propn.extend(_residue_propn(uce))
        records.append(
            {
                "id": uce.get("id"),
                "words": words,
                "base": base,
                "rescued": rescued,
                "rescued_tool": rescued_tool,
                "propn": propn,
            }
        )
    return records


# ──────────────────────────────────────────────────────────────────────
# Modo 1: tabla de tokens exacta
# ----------------------------------------------------------------------
def build_exact_records(table: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    content_pos = {policy: set(policy_content_pos(policy)) for policy in POLICIES}
    records: List[Dict[str, Any]] = []

    for uce_id, entry in table.items():
        rows = entry.get("tokens") or []
        words = entry.get("words")
        if not words:
            words = len(rows)

        active = {policy: [] for policy in POLICIES}
        rescued = 0
        rescued_tool = 0
        for row in rows:
            pos = row.get("pos")
            lemma = row.get("lemma") or row.get("forma") or ""
            is_stop = bool(row.get("is_stop"))
            is_alpha = bool(row.get("is_alpha"))
            form = fold(row.get("forma"))
            decisions = {}
            for policy in POLICIES:
                decisions[policy] = token_decision(
                    policy=policy,
                    pos=pos,
                    lemma=lemma,
                    is_stop=is_stop,
                    is_alpha=is_alpha,
                    content_pos=content_pos[policy],
                )
                if decisions[policy] == "active":
                    active[policy].append(form)

            if decisions["pos_plus_propn"] == "active":
                if decisions["pos_only"] != "active":
                    rescued += 1
                    if is_tool_adverb("tool_adverb_filtered", pos, lemma):
                        rescued_tool += 1

        records.append(
            {
                "id": uce_id,
                "words": words,
                "base": active["pos_only"],
                "rescued": [],
                "rescued_tool": ["__tool__"] * rescued_tool,
                "propn": [],
                "forms_by_policy": active,
                "rescued_n": rescued,
            }
        )
    return records


# ──────────────────────────────────────────────────────────────────────
# Métricas
# ----------------------------------------------------------------------
def _subtract(forms: Sequence[str], remove: Iterable[str]) -> List[str]:
    remaining = Counter(remove)
    kept: List[str] = []
    for form in forms:
        if remaining.get(form, 0) > 0:
            remaining[form] -= 1
        else:
            kept.append(form)
    return kept


def policy_forms(record: Dict[str, Any], policy: str) -> List[str]:
    if "forms_by_policy" in record:
        return record["forms_by_policy"][policy]
    if policy == "pos_only":
        return record["base"]
    rescued = record["rescued"]
    if policy == "tool_adverb_filtered":
        rescued = _subtract(rescued, record["rescued_tool"])
    return record["base"] + rescued + record["propn"]


def _rescue_stats(record: Dict[str, Any], policy: str) -> Tuple[int, int, int]:
    """(añadidas_totales, añadidas_is_stop, herramienta_añadida) por UCE."""
    if policy == "pos_only":
        return 0, 0, 0
    if "forms_by_policy" in record:
        added = len(record["forms_by_policy"][policy]) - len(record["base"])
        isstop = record.get("rescued_n", 0)
        tool = len(record["rescued_tool"])
        if policy == "tool_adverb_filtered":
            return added, isstop - tool, 0
        return added, isstop, tool
    rescued = record["rescued"]
    tool = record["rescued_tool"]
    if policy == "tool_adverb_filtered":
        return (
            len(rescued) - len(tool) + len(record["propn"]),
            len(rescued) - len(tool),
            0,
        )
    return len(rescued) + len(record["propn"]), len(rescued), len(tool)


def measure(
    records: Sequence[Dict[str, Any]], policy: str, bootstrap: int, seed: int
) -> Dict[str, Any]:
    forms_by_uce = [policy_forms(record, policy) for record in records]
    words_by_uce = [record["words"] for record in records]
    rescue_pairs = [_rescue_stats(record, policy) for record in records]

    n = len(records)
    total_lemmas = sum(len(forms) for forms in forms_by_uce)
    total_words = sum(words_by_uce)
    rescued = sum(pair[0] for pair in rescue_pairs)
    rescued_isstop = sum(pair[1] for pair in rescue_pairs)
    rescued_tool = sum(pair[2] for pair in rescue_pairs)

    result = {
        "policy": policy,
        "n": n,
        "lemmas": total_lemmas,
        "words": total_words,
        "lemas_uce": total_lemmas / n if n else 0.0,
        "retention": total_lemmas / total_words if total_words else 0.0,
        "distinct_uce": (
            sum(len(set(forms)) for forms in forms_by_uce) / n if n else 0.0
        ),
        "rescued": rescued,
        "rescued_isstop": rescued_isstop,
        "rescued_tool": rescued_tool,
        "quota": (rescued_tool / rescued) if rescued > 0 else 0.0,
    }
    result["ci"] = _bootstrap_ci(forms_by_uce, words_by_uce, bootstrap, seed)
    return result


def _bootstrap_ci(
    forms_by_uce: List[List[str]], words_by_uce: List[int], bootstrap: int, seed: int
) -> Dict[str, Tuple[float, float]]:
    n = len(forms_by_uce)
    if n == 0 or bootstrap <= 0:
        return {"lemas_uce": (0.0, 0.0), "retention": (0.0, 0.0)}

    rng = random.Random(seed)
    means: List[float] = []
    retentions: List[float] = []
    for _ in range(bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        lemmas = sum(len(forms_by_uce[i]) for i in idx)
        words = sum(words_by_uce[i] for i in idx)
        means.append(lemmas / n)
        retentions.append(lemmas / words if words else 0.0)

    def percentile(values: List[float]) -> Tuple[float, float]:
        values.sort()
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        return lo, hi

    return {"lemas_uce": percentile(means), "retention": percentile(retentions)}


# ──────────────────────────────────────────────────────────────────────
# Regla pre-registrada
# ----------------------------------------------------------------------
def apply_rule(
    results: Sequence[Dict[str, Any]], threshold: float
) -> Optional[Dict[str, Any]]:
    eligible = [r for r in results if r["quota"] <= threshold + 1e-12]
    if not eligible:
        return None
    return min(eligible, key=lambda r: abs(r["retention"] - ALCESTE_RETENTION))


# ──────────────────────────────────────────────────────────────────────
# Presentación
# ----------------------------------------------------------------------
def render(results: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append(
        "POLÍTICAS DE RETENCIÓN DE FORMAS ACTIVAS — medición offline  "
        "(ALCESTE 2012 = 13.95 lemas/UCE / 0.537)"
    )
    lines.append("=" * 100)
    lines.append("fuente        : {}".format(meta["input"]))
    lines.append("modo          : {}".format(meta["mode"]))
    lines.append(
        "corpus        : {} UCEs | {} palabras crudas (texto.split) | "
        "{} tokens alfa".format(meta["n"], meta["words"], meta["tokens"])
    )
    lines.append(
        "bootstrap     : {} remuestreos, seed={}".format(
            meta["bootstrap"], meta["seed"]
        )
    )
    lines.append("")
    header = "{:<36} {:>9} {:>19} {:>9} {:>17} {:>11} {:>8} {:>8} {:>7}".format(
        "política",
        "lemas/UCE",
        "IC95% lemas/UCE",
        "retención",
        "IC95% retención",
        "formas/UCE",
        "rescat.",
        "herr.",
        "cuota",
    )
    lines.append(header)
    lines.append("-" * len(header))
    for result in results:
        ci_lemas = result["ci"]["lemas_uce"]
        ci_ret = result["ci"]["retention"]
        lines.append(
            "{:<36} {:>9.2f} {:>19} {:>9.3f} {:>17} {:>11.2f} {:>8} {:>8} {:>7.3f}".format(
                LABELS[result["policy"]],
                result["lemas_uce"],
                "[{:.2f}, {:.2f}]".format(ci_lemas[0], ci_lemas[1]),
                result["retention"],
                "[{:.3f}, {:.3f}]".format(ci_ret[0], ci_ret[1]),
                result["distinct_uce"],
                result["rescued"],
                result["rescued_tool"],
                result["quota"],
            )
        )
    lines.append(
        "{:<36} {:>9} {:>19} {:>9} {:>17} {:>11} {:>8} {:>8} {:>7}".format(
            "ALCESTE 2012 (referencia)", "13.95", "-", "0.537", "-", "-", "-", "-", "-"
        )
    )
    lines.append("")
    if meta.get("validation"):
        lines.append(meta["validation"])
        lines.append("")
    return "\n".join(lines)


def render_verdict(results: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("REGLA PRE-REGISTRADA (docs/design_v5/RETENTION_POLICY.md §2)")
    lines.append(
        "  [1] restricción dura: cuota de palabras herramienta sobre lo rescatado "
        "<= {:.2f}".format(TOOL_QUOTA_MAX)
    )
    lines.append(
        "  [2] criterio: entre las admitidas, min |retención - {:.3f}|".format(
            ALCESTE_RETENTION
        )
    )
    lines.append("")
    for result in results:
        admitted = result["quota"] <= TOOL_QUOTA_MAX + 1e-12
        gap = abs(result["retention"] - ALCESTE_RETENTION)
        lines.append(
            "  {:<36} cuota={:.3f}  {}   |ret-ALCESTE|={:.3f}".format(
                result["policy"],
                result["quota"],
                "ADMITIDA" if admitted else "EXCLUIDA",
                gap,
            )
        )
    lines.append("")
    winner = apply_rule(results, TOOL_QUOTA_MAX)
    if winner is None:
        lines.append("  VEREDICTO: ninguna política cumple la restricción dura.")
        lines.append(
            "  -> se publica el baseline y la desviación cuantificada "
            "(no se 'arregla' el número)."
        )
    else:
        lines.append("  VEREDICTO: {}".format(LABELS[winner["policy"]]))
    lines.append("")
    lines.append("  Sensibilidad de la restricción dura (mismo criterio [2]):")
    for threshold in QUOTA_SENSITIVITY:
        selected = apply_rule(results, threshold)
        lines.append(
            "    cuota <= {:.2f} -> {}".format(
                threshold,
                LABELS[selected["policy"]] if selected else "ninguna",
            )
        )
    lines.append("")
    lines.append(
        "  Denominadores de la cuota: (a) rescatadas totales = is_stop + PROPN; "
        "(b) sólo is_stop."
    )
    for result in results:
        if result["policy"] == "pos_only":
            continue
        isstop = result.get("rescued_isstop")
        if isstop:
            lines.append(
                "    {:<36} (a) {}/{}={:.3f}  (b) {}/{}={:.3f}".format(
                    result["policy"],
                    result["rescued_tool"],
                    result["rescued"],
                    result["quota"],
                    result["rescued_tool"],
                    isstop,
                    result["rescued_tool"] / isstop,
                )
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/workflow_data.json",
        help="workflow_data.json (528 UCEs)",
    )
    parser.add_argument(
        "--token-table",
        default=None,
        help="JSONL de TokenTableWriter (medición exacta)",
    )
    parser.add_argument(
        "--fingerprint", default="", help="corpus_fingerprint de la tabla de tokens"
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="remuestreos bootstrap (0 = sin IC)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", default=None, help="escribe las métricas a este JSON")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        raise SystemExit("No existe --input: {}".format(args.input))

    uces = load_workflow(args.input)
    words = sum(len((uce.get("texto") or "").split()) for uce in uces)
    tokens = sum(len(uce.get("tokens") or []) for uce in uces)

    table = None
    if args.token_table:
        table = read_token_table(args.token_table, args.fingerprint)
        if table is None:
            raise SystemExit(
                "Tabla de tokens ausente o con huella distinta: {}".format(
                    args.token_table
                )
            )

    if table is not None:
        records = build_exact_records(table)
        mode = "exacto (tabla de tokens)"
    else:
        records = build_residue_records(uces)
        mode = "aproximado (reconstrucción por residuo)"

    results = [
        measure(records, policy, args.bootstrap, args.seed) for policy in POLICIES
    ]

    validation = None
    if len(uces) == PUBLISHED_UCES and table is None:
        checks = []
        ok = True
        for result in results:
            expected = PUBLISHED_COUNTS[result["policy"]]
            same = result["lemmas"] == expected
            ok = ok and same
            checks.append(
                "{}={}{}".format(
                    result["policy"],
                    result["lemmas"],
                    "" if same else " (esperado {})".format(expected),
                )
            )
        validation = "VALIDACIÓN de la reconstrucción: {} | {}".format(
            "VALIDADA" if ok else "FALLA", ", ".join(checks)
        )

    meta = {
        "input": os.path.abspath(args.input),
        "mode": mode,
        "n": len(uces),
        "words": words,
        "tokens": tokens,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "validation": validation,
    }

    print(render(results, meta))
    print(render_verdict(results))

    if args.json:
        payload = {
            "mode": mode,
            "meta": meta,
            "results": results,
            "verdict": (apply_rule(results, TOOL_QUOTA_MAX) or {}).get("policy"),
            "tool_quota_max": TOOL_QUOTA_MAX,
            "alceste": {
                "lemas_uce": ALCESTE_LEMMAS_PER_UCE,
                "retention": ALCESTE_RETENTION,
            },
        }
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
