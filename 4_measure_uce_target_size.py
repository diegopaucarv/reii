#!/usr/bin/env python3
"""
Script 4 — Coste y forma de la segmentación según uce_target_size.

Hechos verificados que cambian el diseño original:
  * main_workflow.py NO tiene --corpus, --pass ni --bench (argv manual: los flags
    desconocidos se ignoraban en silencio y corría el pipeline COMPLETO con el corpus
    por defecto). Se añadió --bench / --bench-coref en el parche.
  * --uce-target-size SÍ existe y desactiva la auto-calibración (calibrar_uce_target_size).
  * El corpus sale de data/txt_outputs/tmp/*.json (nivel de módulo) y se omiten los
    documentos ya en PostgreSQL salvo --force: aquí siempre se pasa --force.
  * La caché de coref está indexada por uce_target_size (`fp:uce_target:min_words`):
    cada tamaño nuevo implica recomputar coref -> ese es el coste real a medir.

Salida: tabla tamaño -> n_UCE, palabras/UCE, tiempos; ajuste lineal coste~n_UCE y
extrapolación; y comparación con ALCESTE 2012 (26 ocurrencias/UCE -> 376 UCE).

Uso:
  python 4_measure_uce_target_size.py --sizes 18 22 26 30 --coref
  python 4_measure_uce_target_size.py --sizes 18 26 --extrapolate 5809 --dry-run
"""

from __future__ import annotations

import argparse
import json

from diag_common import DIAG_DIR, detect_runner, parse_bench, run_to_log, workflow_cmd

ALCESTE_REF = {"uce_size": 26, "n_uce": 376, "words_per_uce_analyzed": 13.95}


def linfit(x, y):
    n = len(x)
    if n < 2 or len(set(x)) < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    return b, my - b * mx


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sizes", type=int, nargs="+", default=[18, 22, 26, 30])
    ap.add_argument(
        "--coref",
        action="store_true",
        help="mide también coref (lento; requiere Stanza)",
    )
    ap.add_argument("--runner", choices=["docker", "local"], default=None)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument(
        "--extrapolate", type=int, default=None, help="nº de UCE de un corpus mayor"
    )
    ap.add_argument("--out", default=str(DIAG_DIR / "uce_size_bench.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    runner = detect_runner(a.runner)
    rows = []
    for s in a.sizes:
        print(f"\n── uce_target_size={s} ──")
        flags = ["--force", "--bench", "--uce-target-size", str(s)] + (
            ["--bench-coref"] if a.coref else []
        )
        text = run_to_log(
            workflow_cmd(runner, *flags),
            DIAG_DIR / f"bench_uce{s}.log",
            a.timeout,
            a.dry_run,
        )
        if a.dry_run or text is None:
            continue
        b = parse_bench(text)
        if "n_uces" not in b:
            print("  ✗ sin líneas [BENCH]: ¿se aplicó el parche a main_workflow.py?")
            continue
        rows.append({"size": s, **b})
        print("  " + ", ".join(f"{k}={v:g}" for k, v in b.items()))
    if not rows:
        return
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)

    print("\n" + "=" * 72)
    print(
        f"{'size':>5} {'n_UCE':>6} {'pal/UCE':>8} {'seg_s':>7} {'lem_s':>7} {'coref_s':>8}"
    )
    for r in rows:
        print(
            f"{r['size']:5d} {r['n_uces']:6.0f} {r['mean_words_per_uce']:8.1f} "
            f"{r.get('seg_seconds', 0):7.1f} {r.get('lemmatize_seconds', 0):7.1f} {r.get('coref_seconds', float('nan')):8.1f}"
        )
    print(
        f"\nALCESTE 2012 (mismo corpus): size={ALCESTE_REF['uce_size']} -> {ALCESTE_REF['n_uce']} UCE."
    )
    near = min(rows, key=lambda r: abs(r["n_uces"] - ALCESTE_REF["n_uce"]))
    print(
        f"Tamaño cuyo nº de UCE se parece más al de ALCESTE: {near['size']} ({near['n_uces']:.0f} UCE)."
    )
    if a.extrapolate:
        for key in ("seg_seconds", "lemmatize_seconds", "coref_seconds"):
            pts = [(r["n_uces"], r[key]) for r in rows if key in r]
            f = linfit([p[0] for p in pts], [p[1] for p in pts])
            if f:
                print(
                    f"  {key}: ~{f[0] * a.extrapolate + f[1]:.0f}s para {a.extrapolate} UCE (ajuste lineal, ≥2 puntos)"
                )
    print(f"\nResultados: {a.out}")


if __name__ == "__main__":
    main()
