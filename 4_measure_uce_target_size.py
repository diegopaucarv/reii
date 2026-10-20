#!/usr/bin/env python3
from __future__ import annotations

"""
Script 4: Mide coste de uce_target_size.
Ejecuta segmentación + coref para 5-7 valores distintos de uce_target_size
y reporta:
  - Segundos de segmentación por valor
  - Segundos de pase de coref (ctx=2) por valor
  - Varianza entre corridas idénticas
  - Estimación de coste de búsqueda: k·(seg + coref·ctx_count)

Uso:
  python 4_measure_uce_target_size.py --reii-dir . --corpus corpus.txt \
    --uce-values "18,22,26,30,34" --repeats 2 --out timing.tsv
"""
import re
import subprocess
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path


def extract_timing(output):
    """Busca en stdout '[BENCH]' líneas con tiempos."""
    times = {}
    for line in output.split("\n"):
        if "[BENCH]" in line or "[COREF]" in line or "[TIEMPO]" in line:
            # Buscar patrones como "seg_time=1.23" o "coref_ctx=2_time=4.56"
            m_seg = re.search(r"seg(?:ment)?.*time[=:]?\s*([\d.]+)", line, re.I)
            m_coref = re.search(r"coref.*time[=:]?\s*([\d.]+)", line, re.I)
            if m_seg:
                times["segmentation"] = float(m_seg.group(1))
            if m_coref:
                times["coref"] = float(m_coref.group(1))
    return times


def run_with_uce_value(reii_dir, corpus_path, uce_target_size, repeat=1):
    """Corre el pipeline con un valor fijo de uce_target_size."""
    reii_dir = Path(reii_dir)
    corpus_path = Path(corpus_path)

    all_times = []
    for rep in range(repeat):
        cmd = [
            sys.executable,
            "-m",
            "reii.main_workflow",
            "--corpus",
            str(corpus_path),
            "--uce-target-size",
            str(uce_target_size),
            "--pass",
            "1",  # Solo etapa barata
            "--bench",  # Si soporta flag de benchmark
        ]

        start = time.time()
        result = subprocess.run(
            cmd,
            cwd=reii_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.time() - start

        times = extract_timing(result.stdout + result.stderr)
        times["total"] = elapsed
        all_times.append(times)

        print(f"  Rep {rep + 1}/{repeat}: {elapsed:.1f}s total", end="")
        if "segmentation" in times:
            print(
                f" (seg={times['segmentation']:.1f}s, coref={times.get('coref', '?')}s)",
                end="",
            )
        print()

    return all_times


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--reii-dir", default=".", help="Ruta a REII")
    parser.add_argument("--corpus", required=True, help="Ruta al corpus")
    parser.add_argument(
        "--uce-values",
        default="18,22,26,30,34",
        help="Valores de uce_target_size (CSV)",
    )
    parser.add_argument("--repeats", type=int, default=2, help="Repeticiones por valor")
    parser.add_argument("--out", default="timing.tsv", help="Fichero de salida")
    args = parser.parse_args()

    uce_values = [int(v.strip()) for v in args.uce_values.split(",")]

    print("=" * 70)
    print("SCRIPT 4: Medir coste de uce_target_size")
    print("=" * 70)
    print(f"Corpus: {args.corpus}")
    print(f"Valores de uce_target_size: {uce_values}")
    print(f"Repeticiones por valor: {args.repeats}\n")

    results = defaultdict(list)
    for uce_val in uce_values:
        print(f"→ uce_target_size={uce_val}:")
        times = run_with_uce_value(args.reii_dir, args.corpus, uce_val, args.repeats)
        results[uce_val] = times

    print("\n" + "=" * 70)
    print("RESUMEN:")
    print(
        f"{'uce_target_size':<15} {'seg (s)':<10} {'coref (s)':<10} {'total (s)':<10} {'var total':<10}"
    )
    print("-" * 60)

    seg_times = {}
    coref_times = {}
    for uce_val in uce_values:
        times_list = results[uce_val]
        seg = [t.get("segmentation", 0) for t in times_list if "segmentation" in t]
        coref = [t.get("coref", 0) for t in times_list if "coref" in t]
        total = [t.get("total", 0) for t in times_list]

        seg_avg = sum(seg) / len(seg) if seg else 0
        coref_avg = sum(coref) / len(coref) if coref else 0
        total_avg = sum(total) / len(total)
        total_var = (
            (sum((t - total_avg) ** 2 for t in total) / len(total)) ** 0.5
            if len(total) > 1
            else 0
        )

        print(
            f"{uce_val:<15} {seg_avg:<10.1f} {coref_avg:<10.1f} {total_avg:<10.1f} {total_var:<10.2f}"
        )
        seg_times[uce_val] = seg_avg
        coref_times[uce_val] = coref_avg

    print("\n" + "=" * 70)
    print("ANÁLISIS:")

    # ¿Crece lineal el coste de seg?
    if len(seg_times) >= 2:
        sorted_vals = sorted(seg_times.items())
        delta_uce = sorted_vals[-1][0] - sorted_vals[0][0]
        delta_seg = sorted_vals[-1][1] - sorted_vals[0][1]
        if delta_seg > 0:
            rate = delta_seg / delta_uce
            print(f"Tasa de crecimiento (segmentación): {rate:.3f}s per uce_unit")

    # ¿El ruido es significativo?
    all_vars = []
    for uce_val in uce_values:
        times_list = results[uce_val]
        total = [t.get("total", 0) for t in times_list]
        if len(total) > 1:
            avg = sum(total) / len(total)
            var = (sum((t - avg) ** 2 for t in total) / len(total)) ** 0.5
            all_vars.append(var)
            if var / avg > 0.05:  # >5% CV
                print(f"⚠ Ruido alto en uce={uce_val}: CV={100 * var / avg:.1f}%")

    if all_vars:
        avg_var = sum(all_vars) / len(all_vars)
        print(f"Ruido promedio: {avg_var:.1f}s (CV={100 * avg_var / 20:.1f}%)")

    # Guardar
    with open(args.out, "w") as fh:
        fh.write("uce_target_size\trepeat\tseg_time\tcoref_time\ttotal_time\n")
        for uce_val in uce_values:
            for rep, times in enumerate(results[uce_val]):
                fh.write(
                    f"{uce_val}\t{rep}\t"
                    f"{times.get('segmentation', 'NA')}\t"
                    f"{times.get('coref', 'NA')}\t"
                    f"{times.get('total', 'NA')}\n"
                )

    print(f"\n→ Datos completos en: {args.out}")
    print("\nPara modelar uce_target_size en el optimizador:")
    print(
        f"  - Coste unitario de segmentación: ~{seg_times.get(sorted(seg_times.keys())[0], 0):.1f}s"
    )
    print(
        f"  - Coste unitario de coref: ~{coref_times.get(sorted(coref_times.keys())[0], 0):.1f}s/ctx"
    )
    print(f"  - Ruido de proceso: ~{all_vars[0] if all_vars else 0:.1f}s")


if __name__ == "__main__":
    main()
