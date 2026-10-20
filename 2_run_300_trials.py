#!/usr/bin/env python3
"""
Script 2: Corre 300 trials del optimizador y ve si converge.
Compara 2-3 semillas de Optuna para medir estabilidad.

NOTA: Este script modifica temporalmente main_workflow.py para cambiar optimize_trials.
Asume que Config() está en main_workflow.py cerca de "optimize_trials=".

Uso:
  python 2_run_300_trials.py --reii-dir /ruta/a/reii --seeds 3 --out results.tsv

Reporta:
  - Mejor utilidad por semilla
  - Estabilidad (σ del óptimo)
  - ¿Se va a un valor fijo o explora?
"""

import json
import re
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path


def patch_optimize_trials(reii_dir, new_value):
    """Modifica optimize_trials en main_workflow.py."""
    mw_path = Path(reii_dir) / "src" / "reii" / "main_workflow.py"
    if not mw_path.exists():
        print(f"✗ No encontré main_workflow.py en {mw_path}")
        return False

    content = mw_path.read_text(encoding="utf-8")
    # Buscar optimize_trials=NNN (cualquier número)
    original = content
    content = re.sub(r"(optimize_trials\s*=\s*)\d+", f"\\1{new_value}", content)

    if content == original:
        print(f"⚠ No encontré optimize_trials en main_workflow.py")
        return False

    mw_path.write_text(content, encoding="utf-8")
    print(f"  ✓ optimize_trials seteado a {new_value}")
    return True


def restore_optimize_trials(reii_dir, original_value):
    """Restaura el valor original de optimize_trials."""
    mw_path = Path(reii_dir) / "src" / "reii" / "main_workflow.py"
    content = mw_path.read_text(encoding="utf-8")
    content = re.sub(r"(optimize_trials\s*=\s*)\d+", f"\\1{original_value}", content)
    mw_path.write_text(content, encoding="utf-8")


def run_optimizer(reii_dir, seed, seed_arg_name="optimize_seed"):
    """Corre Optuna con semilla fija y retorna best_params."""
    reii_dir = Path(reii_dir)

    # Leer main_workflow.py para encontrar dónde setear la semilla
    mw_path = reii_dir / "src" / "reii" / "main_workflow.py"
    content = mw_path.read_text(encoding="utf-8")

    # Buscar donde se setea la semilla de Optuna
    # Posibles nombres: optimize_seed, random_state, sampler_seed, etc.
    possible_names = ["optimize_seed", "random_state", "optuna_seed", "sampler_seed"]
    found = None
    for name in possible_names:
        if name in content:
            found = name
            break

    if found:
        content = re.sub(f"({found}\\s*=\\s*)\\d+", f"\\1{seed}", content)
        mw_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Semilla seteada a {seed} ({found})")
    else:
        print(f"  ⚠ No encontré variable de semilla en main_workflow.py")

    print(f"→ Seed {seed}: Corriendo (esto toma ~30-60 min)...")
    result = subprocess.run(
        [sys.executable, "src/reii/main_workflow.py"],
        cwd=reii_dir,
        capture_output=True,
        text=True,
        timeout=3600 * 4,
    )

    # Buscar best_value/best_params en el output
    output = result.stdout + result.stderr

    # Buscar "best utility: 0.XXX" o "Best params:" en el output
    utility_match = re.search(r"best.*utility[:\s]+([0-9.]+)", output, re.I)
    params_match = re.search(r"Best params[:\s]*\{[^}]+\}", output, re.I)

    utility = None
    params = None
    if utility_match:
        try:
            utility = float(utility_match.group(1))
        except:
            pass

    if params_match:
        try:
            params_str = params_match.group(0)
            params = json.loads(
                params_str.split(":", 1)[1].strip() if ":" in params_str else params_str
            )
        except:
            pass

    return {
        "seed": seed,
        "utility": utility,
        "params": params,
        "output_sample": output[-500:] if output else "",
        "passed": utility is not None,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--reii-dir", required=True, help="Ruta al directorio de REII")
    parser.add_argument(
        "--seeds", type=int, default=3, help="Cuántas semillas (default 3)"
    )
    parser.add_argument(
        "--max-trials", type=int, default=300, help="Trials por semilla"
    )
    parser.add_argument(
        "--out", default="results_300trials.tsv", help="Fichero de salida"
    )
    args = parser.parse_args()

    reii_dir = Path(args.reii_dir)
    if not reii_dir.exists():
        print(f"✗ Directorio REII no existe: {reii_dir}")
        sys.exit(1)

    print("=" * 70)
    print(f"SCRIPT 2: Optimizador con {args.max_trials} trials × {args.seeds} semillas")
    print("=" * 70)
    print(f"REII: {reii_dir}\n")

    # Leer el valor original de optimize_trials
    mw_path = reii_dir / "src" / "reii" / "main_workflow.py"
    original_content = mw_path.read_text(encoding="utf-8")
    match = re.search(r"optimize_trials\s*=\s*(\d+)", original_content)
    original_trials = int(match.group(1)) if match else 100

    try:
        # Setear optimize_trials
        if not patch_optimize_trials(reii_dir, args.max_trials):
            print("✗ No se pudo setear optimize_trials. Verifica main_workflow.py")
            sys.exit(1)

        results = []
        for seed in range(args.seeds):
            print(f"\n[{seed + 1}/{args.seeds}]")
            res = run_optimizer(reii_dir, seed)
            results.append(res)
            if res["passed"]:
                print(f"  ✓ Seed {seed}: utility={res['utility']:.4f}")
                if res["params"]:
                    print(f"    Parámetros: {res['params']}")
            else:
                print(f"  ✗ Seed {seed}: no se encontró best_utility en output")
                if res["output_sample"]:
                    print(f"    Últimas líneas: {res['output_sample']}")

        print("\n" + "=" * 70)
        print("RESUMEN:")
        passed = [r for r in results if r["passed"]]
        if not passed:
            print("✗ Ninguna corrida pasó (no se encontró best_utility en output)")
            print(
                "  Verifica que la salida de main_workflow.py incluya 'best utility: X'"
            )
            sys.exit(1)

        utils = [r["utility"] for r in passed]
        print(f"Utilidades: {[f'{u:.4f}' for u in utils]}")
        print(f"  Media: {sum(utils) / len(utils):.4f}")
        if len(utils) > 1:
            var = sum((u - sum(utils) / len(utils)) ** 2 for u in utils) / len(utils)
            sd = var**0.5
            print(f"  Desv estándar: {sd:.4f}")
        print(f"  Rango: {min(utils):.4f} → {max(utils):.4f}")

        print("\n¿CONVERGE?")
        if max(utils) - min(utils) < 0.01:
            print("  ✓ SÍ: todas las semillas dan resultados muy similares")
        elif max(utils) - min(utils) < 0.05:
            print("  ~ Parcialmente: algo de variación pero razonable")
        else:
            print("  ✗ No: hay mucha variación entre semillas")

        print("\n¿SE VA A UN VALOR FIJO?")
        param_sets = [
            json.dumps(r["params"], sort_keys=True) for r in passed if r["params"]
        ]
        if param_sets:
            unique_params = set(param_sets)
            if len(unique_params) == 1:
                print("  ✓ SÍ: todos convergen al mismo parámetro")
                print(f"    {passed[0]['params']}")
            else:
                print(
                    f"  ~ NO: {len(unique_params)} parámetros distintos entre {len(param_sets)}"
                )
        else:
            print("  ⚠ No se extrajeron parámetros del output")

        # Guardar
        with open(args.out, "w") as fh:
            fh.write(f"# Corrida {datetime.now()}\n")
            fh.write(f"# max_trials={args.max_trials}\n")
            fh.write("seed\tutility\tpassed\tparams\n")
            for r in results:
                fh.write(
                    f"{r['seed']}\t"
                    f"{r['utility'] or 'NA'}\t"
                    f"{r['passed']}\t"
                    f"{json.dumps(r['params']) if r['params'] else 'NA'}\n"
                )

        print(f"\n→ Resultados guardados en: {args.out}")

    finally:
        # Restaurar
        restore_optimize_trials(reii_dir, original_trials)
        print(f"  ✓ Restaurado optimize_trials a {original_trials}")


if __name__ == "__main__":
    main()
