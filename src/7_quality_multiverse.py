#!/usr/bin/env python3
"""
Script 7 - Multiverso de especificacion: curva de resultados por corrida.

Pregunta que responde: con el MISMO corpus y el MISMO codigo, al cambiar
uce_target_size, cambia el numero de clases (K), el reparto y la retencion? Si
cambia, las clases no estan identificadas: el "resultado" es un artefacto de la
especificacion, no del corpus.

Fuente de datos (NO se recalcula nada, solo se lee):
  * Tablas Postgres quality_run / quality_uce, que el pipeline YA escribe en
    cada corrida via reii.quality.hooks.save_quality_run.
  * run_id = "run-" + sha1(corpus_hash|code_version|seed|mode|config)[:16],
    asi que cada uce_target_size es una corrida DISTINTA dentro de la misma BD.

REGLA CENTRAL (y error que esta version corrige):
  Comparar K entre corridas solo tiene sentido si comparten TODO el config
  salvo el target. Por eso cada corrida se agrupa por
      (firma del config sin claves de target/seed, mode)
  y la curva de especificacion se dibuja DENTRO del grupo. Mezclar configs con
  hiperparametros distintos produce K "inestable" que es puro artefacto.

OJO con corpus_hash: NO sirve para agrupar. save_quality_run lo calcula sobre
el set de UCEs YA SEGMENTADO, asi que cambia al cambiar uce_target_size (tres
corridas del mismo corpus con target 18/22/26 dan tres corpus_hash distintos).
Se muestra en la tabla como informacion, nunca como clave.

Unidades (no mezclar):
  * palabras crudas/UCE : lo que mide el segmentador (script 4). target ~ 26.
  * palabras ANALIZADAS/UCE : 13.95 en ALCESTE, y aqui = lemas/UCE (n_tokens).
  * retencion = lemas/UCE / palabras crudas/UCE. ALCESTE: 13.95/26 = 0.536.
    Si estas por debajo, filtramos mas tokens por UCE; por encima, entran
    palabras vacias. NO se compara lemas con "ocurrencias" ni con el 68 % de
    cobertura de ALCESTE (eso es cobertura de TEXTO, otra cosa).

Otras correcciones respecto de la version anterior:
  * TV se calcula sobre el VECTOR DE TAMANOS DE CLASE ordenado (permutacion
    invariante). Con indices de clase crudos daba 1.000 entre corridas con el
    mismo reparto, porque los ids de cluster son arbitrarios.
  * Si una corrida tiene quality_run.n_uces_total != filas reales, se avisa.
  * La cobertura solo se calcula si se pasa --corpus-words Y no supera 100 %.

Uso (dentro del contenedor; necesita data/ montado, no la raiz del repo):
  docker compose --profile batch run --rm workflow python -u src/7_quality_multiverse.py
  ... --corpus-words 14331 --out data/diag/quality_multiverse.json
  ... --dsn sqlite:////app/data/diag/fixture.db       # offline / pruebas
  ... --kind pipeline --limit 10

Salida: tabla ASCII + JSON en --out. Sin emojis ni box-drawing (cp1252).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ALCESTE = {
    "n_uce": 376,
    "raw_words_per_uce": 26.0,  # ocurrencias/UCE
    "analyzed_words_per_uce": 13.95,  # palabras analizadas/UCE
}
ALCESTE_RETENTION = ALCESTE["analyzed_words_per_uce"] / ALCESTE["raw_words_per_uce"]
TARGET_KEYS = ("uce_target_size", "uce_target", "target_size", "uce_size")
DROP_TOKENS = ("target", "seed")  # en la firma de especificacion


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def find_target(config: Dict[str, Any]) -> Optional[int]:
    """uce_target_size efectivo de la corrida. Nunca lo inventa."""
    for key in TARGET_KEYS:
        value = config.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    stack = [config]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if (
                    "target" in str(key).lower()
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    return int(value)
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(n for n in node if isinstance(n, (dict, list)))
    return None


def spec_signature(config: Dict[str, Any]) -> str:
    """Firma del config ignorando target y seed: lo que define la especificacion."""

    def strip(node):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                lowered = str(key).lower()
                if any(token in lowered for token in DROP_TOKENS):
                    continue
                out[str(key)] = strip(value)
            return out
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    canon = json.dumps(strip(config or {}), sort_keys=True, default=str)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:8]


def class_sizes(assignments) -> Dict[int, int]:
    if assignments.empty or "cluster_core" not in assignments:
        return {}
    counts: Dict[int, int] = {}
    for value, count in assignments["cluster_core"].dropna().value_counts().items():
        counts[int(value)] = int(count)
    return dict(sorted(counts.items()))


def tv_sorted(a: Dict[int, int], b: Dict[int, int]) -> Optional[float]:
    """TV entre repartos, invariante a la permutacion de etiquetas de clase."""
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return None
    va = [c / na for c in sorted(a.values(), reverse=True)]
    vb = [c / nb for c in sorted(b.values(), reverse=True)]
    width = max(len(va), len(vb))
    va += [0.0] * (width - len(va))
    vb += [0.0] * (width - len(vb))
    return 0.5 * sum(abs(x - y) for x, y in zip(va, vb))


def load_raw_words(bench_json: Path) -> Dict[int, float]:
    """target -> palabras CRUDAS/UCE, medidas por 4_measure_uce_target_size.py."""
    if not bench_json.exists():
        return {}
    try:
        rows = json.loads(bench_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for row in rows:
        size = row.get("size")
        mean = row.get("mean_words_per_uce")
        if size is not None and mean:
            out[int(size)] = float(mean)
    return out


def summarize_run(
    store, run_row: Dict[str, Any], raw_words: Dict[int, float]
) -> Dict[str, Any]:
    run_id = run_row["run_id"]
    detail = store.get_run(run_id) or {}
    config = detail.get("config") or {}
    df = store.load_assignments(run_id)

    # Seleccion de metodo persistida con la corrida (modo classification_mode=select).
    # Vive en summary_json.extras.method_selection (ver reii.quality.hooks).
    _extras = (detail.get("summary") or {}).get("extras") or {}
    _sel = (_extras.get("method_selection") or {}) if isinstance(_extras, dict) else {}

    n_rows = int(len(df))
    sizes = class_sizes(df)
    stable = int(df["cluster_core"].notna().sum()) if n_rows else 0
    tokens = df["n_tokens"].dropna() if n_rows and "n_tokens" in df else None
    lemmas_total = int(tokens.sum()) if tokens is not None and len(tokens) else 0
    lem_per_uce = float(tokens.mean()) if tokens is not None and len(tokens) else None
    s_ac = df["s_acuerdo"].dropna() if n_rows and "s_acuerdo" in df else None
    target = find_target(config)

    retention = None
    if lem_per_uce is not None and target is not None and target in raw_words:
        if raw_words[target]:
            retention = lem_per_uce / raw_words[target]

    return {
        "run_id": run_id,
        "created_at": str(run_row.get("created_at") or ""),
        "corpus_hash": run_row.get("corpus_hash"),
        "spec": spec_signature(config),
        "mode": run_row.get("mode"),
        "seed": run_row.get("seed"),
        "target": target,
        "n_uces_total": int(run_row.get("n_uces_total") or 0),
        "n_rows": n_rows,
        "k": len(sizes),
        "class_sizes": sizes,
        "stable": stable,
        "stable_pct": (100.0 * stable / n_rows) if n_rows else None,
        "lemmas_per_uce": lem_per_uce,
        "lemmas_total": lemmas_total,
        "retention": retention,
        "s_acuerdo_mean": float(s_ac.mean())
        if s_ac is not None and len(s_ac)
        else None,
        # ── metodo elegido (select) ──────────────────────────────
        "selected_method": _sel.get("winner"),
        "selected_k": _sel.get("winner_k"),
        "selected_ari": _sel.get("winner_ari"),
        "selected_coverage": _sel.get("winner_coverage"),
        "selection_ambiguous": bool(_sel.get("ambiguous")),
        "selection_conflict": bool(_sel.get("criteria_conflict")),
    }


def _fmt(value, width, precision=1):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:{width}.{precision}f}"


def print_table(rows: List[Dict[str, Any]], corpus_words: Optional[int]) -> None:
    header = (
        f"{'run':<10} {'corp':>6} {'spec':>6} {'target':>6} {'K':>3} {'UCEs':>6} "
        f"{'stable%':>8} {'lem/UCE':>8} {'lem_total':>10} {'ret':>5} {'s_ac':>6} "
        f"{'met':>4}  flags"
    )
    print("\n" + "=" * 112)
    print(header)
    print("-" * 112)
    for r in rows:
        flags = []
        if r.get("selection_ambiguous"):
            flags.append("casi-empate")
        if r.get("selection_conflict"):
            flags.append("conflicto")
        print(
            f"{r['run_id'][-8:]:<10} "
            f"{str(r['corpus_hash'] or 'n/a')[:6]:>6} {r['spec']:>6} "
            f"{(str(r['target']) if r['target'] is not None else 'n/a'):>6} "
            f"{r['k']:>3} {r['n_rows']:>6} "
            f"{_fmt(r['stable_pct'], 8)} {_fmt(r['lemmas_per_uce'], 8)} "
            f"{r['lemmas_total']:>10} {_fmt(r['retention'], 5, 3)} "
            f"{_fmt(r['s_acuerdo_mean'], 6, 3)} "
            f"{str(r.get('selected_method') or '-'):>4}  "
            f"{', '.join(flags) or '-'}"
        )
    print("=" * 112)
    print("  met = metodo elegido y persistido (A/B/C; '-' = corrida previa a select).")
    print("  ret = lemas/UCE dividido por palabras CRUDAS/UCE medidas (script 4).")
    print(
        f"        ALCESTE: {ALCESTE['analyzed_words_per_uce']}/"
        f"{ALCESTE['raw_words_per_uce']:.0f} = {ALCESTE_RETENTION:.3f}"
    )
    mismatched = [
        r for r in rows if r["n_uces_total"] and r["n_uces_total"] != r["n_rows"]
    ]
    for r in mismatched:
        print(
            f"  [!] {r['run_id'][-8:]}: quality_run.n_uces_total={r['n_uces_total']} "
            f"!= filas en quality_uce={r['n_rows']} (se usa filas)"
        )
    if corpus_words:
        over = [r for r in rows if r["lemmas_total"] > corpus_words]
        if over:
            print(
                f"  [!] {len(over)} corrida(s) superan --corpus-words={corpus_words}: "
                "son de OTRO corpus; no se calcula cobertura."
            )


def group_runs(rows: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    """Agrupa por especificacion (config sin target/seed). NO por corpus_hash."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["spec"], row["mode"]), []).append(row)
    return groups


def print_group(key: tuple, rows: List[Dict[str, Any]]) -> None:
    spec, mode = key
    targets = sorted({r["target"] for r in rows if r["target"] is not None})
    print(
        f"\n== grupo spec={spec} mode={mode or 'n/a'} "
        f"({len(rows)} corrida(s), targets={targets or 'n/a'}) =="
    )
    if len(rows) == 1:
        print("  [!] una sola corrida: no hay curva de especificacion en este grupo.")
        return

    totals = [r["lemmas_total"] for r in rows if r["lemmas_total"]]
    if totals and max(totals) > 3 * min(totals):
        print(
            f"  [!] lem_total varia x{max(totals) / min(totals):.1f} dentro del grupo: "
            "probablemente son corpus distintos con el mismo config. "
            "Curva omitida (comparar corpus diferentes no es valido)."
        )
        return

    by_target: Dict[Optional[int], List[Dict[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(row["target"], []).append(row)

    # ── Estabilidad de la ELECCION de metodo dentro del grupo ────────────
    # Si el ganador cambia al mover el target, la eleccion no es
    # identificable: depende de la segmentacion, no de los datos.
    winners = [r.get("selected_method") for r in rows if r.get("selected_method")]
    if winners:
        distinct = sorted(set(winners))
        if len(distinct) > 1:
            by_t = {
                str(r.get("target")): r.get("selected_method")
                for r in sorted(rows, key=lambda x: (x["target"] is None, x["target"]))
                if r.get("selected_method")
            }
            print(
                f"  [!] el metodo elegido CAMBIA dentro del grupo: "
                f"{by_t} -> la eleccion no es identificable con esta segmentacion"
            )
        else:
            print(f"  [ok] metodo elegido constante: {distinct[0]}")

    print(
        f"  {'target':>6} {'n':>2} {'K':>7} {'stable%':>8} {'lem/UCE':>8} {'ret':>5} {'s_ac':>6}"
    )
    order = sorted(by_target, key=lambda t: (t is None, t))
    for target in order:
        bucket = by_target[target]
        ks = sorted({r["k"] for r in bucket})
        stable = [r["stable_pct"] for r in bucket if r["stable_pct"] is not None]
        lem = [r["lemmas_per_uce"] for r in bucket if r["lemmas_per_uce"] is not None]
        ret = [r["retention"] for r in bucket if r["retention"] is not None]
        sac = [r["s_acuerdo_mean"] for r in bucket if r["s_acuerdo_mean"] is not None]
        k_txt = f"{ks[0]}" if len(ks) == 1 else f"{ks[0]}-{ks[-1]}"
        print(
            f"  {str(target):>6} {len(bucket):>2} {k_txt:>7} "
            f"{_fmt(sum(stable) / len(stable) if stable else None, 8)} "
            f"{_fmt(sum(lem) / len(lem) if lem else None, 8)} "
            f"{_fmt(sum(ret) / len(ret) if ret else None, 5, 3)} "
            f"{_fmt(sum(sac) / len(sac) if sac else None, 6, 3)}"
        )

    # 1) Ruido en la MISMA especificacion y el MISMO target
    for target, bucket in by_target.items():
        if len(bucket) < 2:
            continue
        ks = [r["k"] for r in bucket]
        stable = [r["stable_pct"] for r in bucket if r["stable_pct"] is not None]
        spread = max(stable) - min(stable) if len(stable) > 1 else 0.0
        print(
            f"  [!] {len(bucket)} replicas con target={target} y MISMA especificacion: "
            f"K={sorted(set(ks))}, rango stable%={spread:.1f} puntos."
        )
        pairs = [(a, b) for i, a in enumerate(bucket) for b in bucket[i + 1 :]]
        tvs = [tv_sorted(a["class_sizes"], b["class_sizes"]) for a, b in pairs]
        tvs = [t for t in tvs if t is not None]
        if tvs:
            print(
                f"      TV entre replicas (invariante a etiquetas): "
                f"{sum(tvs) / len(tvs):.3f} -> {'clases NO reproducibles' if sum(tvs) / len(tvs) > 0.25 else 'reparto reproducible'}"
            )

    # 2) Sensibilidad al target, agregada por target
    if len(targets) >= 2:
        print(
            "  curva target -> K, stable%, lem/UCE, TV (invariante) vs target anterior:"
        )
        previous = None
        for target in order:
            bucket = by_target[target]
            k_vals = sorted({r["k"] for r in bucket})
            stable = [r["stable_pct"] for r in bucket if r["stable_pct"] is not None]
            lem = [
                r["lemmas_per_uce"] for r in bucket if r["lemmas_per_uce"] is not None
            ]
            tv = None
            if previous is not None:
                tvs = [
                    tv_sorted(a["class_sizes"], b["class_sizes"])
                    for a in previous
                    for b in bucket
                ]
                tvs = [t for t in tvs if t is not None]
                tv = sum(tvs) / len(tvs) if tvs else None
            print(
                f"    target={str(target):>4}  K={'-'.join(str(k) for k in k_vals):>5}  "
                f"stable={_fmt(sum(stable) / len(stable) if stable else None, 5)}%  "
                f"lem/UCE={_fmt(sum(lem) / len(lem) if lem else None, 5)}  "
                f"TV={'n/a' if tv is None else f'{tv:.3f}'}"
            )
            previous = bucket

        all_ks = sorted({r["k"] for r in rows})
        if len(all_ks) == 1:
            print(
                f"  [ok] K constante ({all_ks[0]}) en los {len(targets)} targets: "
                "el tamano de UCE NO cambia las clases."
            )
        else:
            print(
                f"  [!] K varia ({all_ks[0]}-{all_ks[-1]}) al cambiar target: "
                "las clases no estan identificadas respecto de la especificacion."
            )
    else:
        print("  [!] un solo target en el grupo: sin curva de especificacion.")


def print_cross_group_note(rows: List[Dict[str, Any]]) -> None:
    """Misma segmentacion, mismo target, distinto K -> el culpable es el clasificador."""
    by_key: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        if row["target"] is None:
            continue
        by_key.setdefault(
            (row["corpus_hash"], row["target"], row["lemmas_total"]), []
        ).append(row)
    hits = [v for v in by_key.values() if len(v) > 1 and len({r["k"] for r in v}) > 1]
    if not hits:
        return
    print(
        "\n[hallazgo] misma segmentacion (mismo corpus, target y lem_total) con K distinto:"
    )
    for bucket in hits:
        ks = sorted(r["k"] for r in bucket)
        specs = sorted({r["spec"] for r in bucket})
        print(
            f"  target={bucket[0]['target']} lem_total={bucket[0]['lemmas_total']}: "
            f"K={ks} (specs {specs})"
        )
    print(
        "  => la segmentacion esta fija y el numero de clases cambia con los "
        "hiperparametros del CLASIFICADOR (o con la semilla)."
    )
    print(
        "     No es un problema de uce_target_size; es de estabilidad del clustering."
    )


def print_diagnostics(rows: List[Dict[str, Any]]) -> None:
    print("\n[diagnostico de unidades]")
    lem = [r["lemmas_per_uce"] for r in rows if r["lemmas_per_uce"] is not None]
    if lem:
        mean = sum(lem) / len(lem)
        print(
            f"  lemas/UCE medio: {mean:.1f}  (ALCESTE: "
            f"{ALCESTE['analyzed_words_per_uce']} palabras ANALIZADAS/UCE)"
        )
    ret = [r["retention"] for r in rows if r["retention"] is not None]
    if ret:
        mean_ret = sum(ret) / len(ret)
        print(
            f"  retencion media lemas/crudas: {mean_ret:.3f}  (ALCESTE {ALCESTE_RETENTION:.3f})"
        )
        if mean_ret < ALCESTE_RETENTION - 0.05:
            print(
                "  [!] filtramos MAS tokens por UCE que ALCESTE: misma UCE cruda, "
                "menos contenido analitico."
            )
        elif mean_ret > ALCESTE_RETENTION + 0.05:
            print("  [!] retenemos MAS tokens: probablemente entran palabras vacias.")
        else:
            print("  [ok] retencion comparable a ALCESTE.")
    else:
        print(
            "  [i] sin retencion calculable: falta data/diag/uce_size_bench.json "
            "o el target no esta en el config."
        )
    print(
        "  [i] esto NO es cobertura de texto: mide que fraccion de los lemas de cada "
        "UCE sobrevive al filtro, no cuanto texto del corpus entra."
    )


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dsn",
        default=None,
        help="override del DSN (por defecto reii.config.DATABASE_URL)",
    )
    ap.add_argument(
        "--corpus-words",
        type=int,
        default=None,
        help="solo para un unico corpus; se ignora si alguna corrida lo supera",
    )
    ap.add_argument("--kind", default=None, help="filtra por quality_run.kind")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--bench-json",
        default="data/diag/uce_size_bench.json",
        help="tabla target -> palabras CRUDAS/UCE (script 4)",
    )
    ap.add_argument("--out", default="data/diag/quality_multiverse.json")
    a = ap.parse_args()

    from reii.quality.store import QualityStore

    store = QualityStore.open(a.dsn)
    runs = store.list_runs(kind=a.kind)
    if runs is None or len(runs) == 0:
        print("[X] quality_run esta vacia: ninguna corrida persistida.")
        print("    save_quality_run se llama al FINAL del pipeline; una corrida")
        print("    interrumpida no aparece. Lanza una corrida completa primero.")
        store.close()
        return 1

    records = runs.to_dict(orient="records")
    records.sort(key=lambda r: str(r.get("created_at") or ""))
    if a.limit:
        records = records[-a.limit :]

    raw_words = load_raw_words(Path(a.bench_json))
    print(f"corridas en quality_run: {len(runs)} (analizando {len(records)})")
    if raw_words:
        print(
            "  palabras crudas/UCE medidas: "
            + ", ".join(f"{k}->{v:.1f}" for k, v in sorted(raw_words.items()))
        )
    else:
        print(f"  [!] sin {a.bench_json}: no se calcula retencion")

    rows = []
    for rec in records:
        try:
            rows.append(summarize_run(store, rec, raw_words))
        except Exception as exc:  # una corrida corrupta no debe tumbar el informe
            print(f"  [!] {rec.get('run_id')}: no se pudo resumir ({exc})")
    store.close()
    if not rows:
        print("[X] ninguna corrida analizable.")
        return 1

    print_table(rows, a.corpus_words)
    for key, bucket in sorted(group_runs(rows).items(), key=lambda kv: -len(kv[1])):
        print_group(key, bucket)
    print_cross_group_note(rows)
    print_diagnostics(rows)

    corpora = {r["corpus_hash"] for r in rows}
    if len(corpora) > 1:
        print(
            f"\n[i] {len(corpora)} valores de corpus_hash: esperado, cambia con la "
            "segmentacion (no identifica corpus). Se agrupa por spec."
        )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_words": a.corpus_words,
        "alceste": {**ALCESTE, "retention": ALCESTE_RETENTION},
        "raw_words_by_target": raw_words,
        "runs": rows,
        "generated_by": "src/7_quality_multiverse.py",
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
