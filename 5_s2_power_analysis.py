#!/usr/bin/env python3
"""
Script 5 (OPCIONAL) — Potencia estadística S2 vs S1 sobre el benchmark MOCK.

Aclaración importante: tests/reii_optimizer_report.py NO mide tu corpus. Simula el
TPESampler de Optuna sobre paisajes sintéticos para comparar dos estrategias de
búsqueda (S1 = fix de lookup; S2 = + factibilidad exacta por pares). Sirve para decidir
una mejora de INGENIERÍA del optimizador; no explica por qué tus clases salen mal.

Correcciones:
  * Ruta por defecto: tests/reii_optimizer_report.py (antes: /mnt/user-data/outputs/...,
    ruta de otra sesión que no existe en tu máquina).
  * El benchmark escribe un .json junto al --out; el script viejo buscaba JSON en
    stdout con un regex y nunca lo encontraba.
  * Añade --skip-tests (si no, corre también la batería de unit tests) y --scenarios.
  * La condición "script 2 converge" no tiene relación lógica con este benchmark: se quitó.
  * Convierte tipos numpy antes de json.dump (np.int64 rompía la serialización).
  * Tiempo: ~65 s por paisaje y escenario -> 32 semillas x 3 escenarios ≈ 105 min.

Uso:
  python 5_s2_power_analysis.py --seeds 32 --scenarios 30:0.7,100:0.7,30:0.3
"""

from __future__ import annotations

import json
import subprocess
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
KEY_REF = "T=coste90_con_fix|S1 fix lookup"
KEY_S2 = "T=coste90_con_fix|S2 +factible exacto (pares)"


def run_benchmark(script, seeds, scenarios, out_md, timeout):
    cmd = [
        sys.executable,
        str(script),
        "--workdir",
        str(REPO),
        "--skip-tests",
        "--seeds",
        str(seeds),
        "--scenarios",
        scenarios,
        "--out",
        str(out_md),
    ]
    print("→ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print("✗ Benchmark falló\n" + r.stderr[-800:])
        return None
    return Path(out_md).with_suffix(".json")


def main():
    ap = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--bench-script", default=str(REPO / "tests" / "reii_optimizer_report.py")
    )
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--scenarios", default="30:0.7,100:0.7,30:0.3")
    ap.add_argument("--out", default="s2_results.json")
    ap.add_argument("--bench-out", default="bench_output.md")
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument(
        "--reuse-json", default=None, help="analiza un .json ya generado sin re-correr"
    )
    a = ap.parse_args()

    if a.reuse_json:
        js = Path(a.reuse_json)
    else:
        if not Path(a.bench_script).exists():
            sys.exit(f"✗ No existe {a.bench_script}")
        js = run_benchmark(a.bench_script, a.seeds, a.scenarios, a.bench_out, a.timeout)
    if not js or not js.exists():
        sys.exit("✗ No se generó el JSON del benchmark")
    data = json.loads(js.read_text(encoding="utf-8")).get("benchmark", {})
    if not data:
        sys.exit("✗ JSON sin sección 'benchmark'")

    res = {}
    for sc, bd in data.items():
        if KEY_REF not in bd or KEY_S2 not in bd:
            print(f"{sc}: faltan claves {KEY_REF!r}/{KEY_S2!r}")
            continue
        d = np.array([r[0] for r in bd[KEY_S2]]) - np.array([r[0] for r in bd[KEY_REF]])
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        sig = bool(len(d) > 1 and d.mean() - ci > 0)
        print(
            f"\n{sc}: Δ(S2-S1)={d.mean():+.4f} ± {ci:.4f}  gana {int((d > 0).sum())}/{len(d)}  significativo={sig}"
        )
        res[sc] = {
            "delta": float(d.mean()),
            "ci": float(ci),
            "wins": int((d > 0).sum()),
            "n": int(len(d)),
            "significant": sig,
        }
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    sigs = [r["significant"] for r in res.values()]
    print(
        "\nRECOMENDACIÓN:",
        "adopta S2"
        if sigs and all(sigs)
        else "mixto: decide por coste/complejidad"
        if any(sigs)
        else "sin evidencia: queda S1",
    )


if __name__ == "__main__":
    main()
