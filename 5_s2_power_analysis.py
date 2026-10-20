#!/usr/bin/env python3
from __future__ import annotations

"""
Script 5 (OPCIONAL): Análisis de potencia para S2.
Re-corre el benchmark con más semillas (16-32 en lugar de 8) para decidir
si S2 es significativamente mejor que S1.

SOLO lo ejecutes si:
  1. El script 2 muestra que sí converge con 300 trials
  2. Quieres gastar tiempo en decidir S2

Uso:
  python 5_s2_power_analysis.py --bench-script /path/to/reii_optimizer_report.py \
    --seeds 32 --out s2_results.json
"""
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def run_benchmark(script_path, seeds=32):
    """Ejecuta el script de benchmark con N semillas."""
    cmd = [
        sys.executable,
        str(script_path),
        "--workdir",
        ".",
        "--seeds",
        str(seeds),
        "--out",
        "bench_output.md",
    ]

    print(f"→ Corriendo benchmark con {seeds} semillas...")
    print(f"  Comando: {' '.join(cmd)}")
    print("  Esto puede tomar 30+ minutos...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        print("✗ Benchmark falló")
        print("STDERR:", result.stderr[-500:])
        return None

    return result.stdout


def analyze_results(output):
    """Extrae resultados del output de benchmark."""
    # El output tiene JSON al final
    try:
        # Buscar el último JSON en el output
        import re

        match = re.search(r'\{.*"benchmark".*\}', output, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data
    except:
        pass
    return None


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-script",
        default="/mnt/user-data/outputs/reii_optimizer_report.py",
        help="Ruta al script de benchmark",
    )
    parser.add_argument("--seeds", type=int, default=32, help="Número de semillas")
    parser.add_argument("--out", default="s2_results.json", help="Salida")
    args = parser.parse_args()

    print("=" * 70)
    print("SCRIPT 5 (OPCIONAL): Análisis de potencia para S2")
    print("=" * 70)

    # Verificar que el script existe
    bench_script = Path(args.bench_script)
    if not bench_script.exists():
        print(f"✗ Script de benchmark no existe: {bench_script}")
        print(f"  Descargalo: https://... o usa reii_optimizer_report.py")
        sys.exit(1)

    output = run_benchmark(bench_script, args.seeds)
    if not output:
        sys.exit(1)

    data = analyze_results(output)
    if not data:
        print("✗ No se pudo extraer resultados del benchmark")
        print("Output:", output[-1000:])
        sys.exit(1)

    print("\n" + "=" * 70)
    print("ANÁLISIS CON N=32 SEMILLAS:")

    # Re-analizar
    res = {}
    for sc, bench_data in data.get("benchmark", {}).items():
        print(f"\n{sc}:")
        # Régimen C (más ajustado)
        ref = np.array([r[0] for r in bench_data[f"T=coste90_con_fix|S1 fix lookup"]])
        s2 = np.array(
            [r[0] for r in bench_data[f"T=coste90_con_fix|S2 +factible exacto (pares)"]]
        )

        d = s2 - ref
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))

        print(f"  S2 vs S1 en régimen C (más información):")
        print(f"    Δ = {d.mean():+.4f} ± {ci:.4f}")
        print(f"    Gana: {(d > 0).sum()}/{len(d)} paisajes")

        sig = "SÍ" if ci > 0 and d.mean() - ci > 0 else "NO"
        print(f"    ¿Significativo (IC no cruza 0)? {sig}")

        res[sc] = {"delta": d.mean(), "ci": ci, "wins": (d > 0).sum(), "n": len(d)}

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"\n→ Resultados en: {args.out}")
    print("\nRECOMENDACIÓN:")
    all_sig = all(r["ci"] > 0 and r["delta"] - r["ci"] > 0 for r in res.values())
    if all_sig:
        print("  ✓ S2 es significativo en TODOS los escenarios → adopta S2")
    else:
        any_sig = any(r["ci"] > 0 and r["delta"] - r["ci"] > 0 for r in res.values())
        if any_sig:
            print(
                "  ~ S2 es significativo en ALGUNOS escenarios → usa criterios de negocio"
            )
        else:
            print("  ✗ S2 no es significativo → NO lo adoptes, usa S1")


if __name__ == "__main__":
    main()
