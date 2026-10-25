#!/usr/bin/env python3
"""
Script 2 — Estabilidad del optimizador entre semillas.

Pregunta: con distinto seed de TPE, ¿el optimizador llega a los mismos parámetros
(punto fijo) o explora? ¿Los ganadores caen en los bordes del espacio?

Qué cambió respecto a la versión anterior (todo verificado en main_workflow.py):
  1. No parchea el código fuente con regex. El viejo regex `random_state\\s*=\\s*\\d+`
     reescribía también randomized_svd (l.2613) y PCA (l.4961) y NO tocaba el campo
     `random_state: int = 42` del dataclass. Ahora usa --seed/--trials/--study-name.
  2. No ejecuta todo el pipeline: usa --optimize-only (se detiene tras el optimizador).
  3. Mueve best_params.json a un lado: si existe, ejecutar() SALTA el optimizador.
  4. Cada semilla usa su propio study (sqlite) -> sin warm-start cruzado.
  5. Lee el resultado de la línea [OPTIMIZER-RESULT] y de best_params.json; el viejo
     regex buscaba "best utility", texto que no existe en el código.
  6. Va por Docker (como tu comando habitual), con --force.

Uso:
  python 2_run_300_trials.py --seeds 0 1 2 --trials 300
  python 2_run_300_trials.py --analyze-only        # re-analiza data/diag/best_params_seed*.json
  python 2_run_300_trials.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
from pathlib import Path

from diag_common import (
    BEST_PARAMS,
    DIAG_DIR,
    at_boundary,
    detect_runner,
    parse_optimizer_results,
    run_to_log,
    workflow_cmd,
)


def run_seed(seed, trials, runner, timeout, dry):
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    study = f"reii_search_v2_seed{seed}"
    cmd = workflow_cmd(
        runner,
        "--force",
        "--optimize-only",
        "--trials",
        str(trials),
        "--seed",
        str(seed),
        "--study-name",
        study,
    )
    text = run_to_log(cmd, DIAG_DIR / f"seed{seed}.log", timeout, dry)
    if dry:
        return None
    if text is None:
        return None
    res = parse_optimizer_results(text)
    if not BEST_PARAMS.exists():
        print(f"  ✗ no se generó {BEST_PARAMS}")
        return None
    dest = DIAG_DIR / f"best_params_seed{seed}.json"
    shutil.copy(BEST_PARAMS, dest)
    if res:
        print(
            f"  ✓ etapa final={res[-1]['stage_mode']} trial=#{res[-1]['trial']} values={res[-1]['values']}"
        )
    return dest


def analyze(files):
    rows = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    if not rows:
        print("Sin resultados que analizar.")
        return
    print("\n" + "=" * 70 + f"\nANÁLISIS ({len(rows)} semillas)\n" + "=" * 70)
    params = sorted({k for r in rows for k in r["params"]})
    print(f"{'parámetro':38s} {'valores por semilla':34s} borde  ¿fijo?")
    for k in params:
        vals = [r["params"].get(k) for r in rows]
        edge = sum(at_boundary(k, v) for v in vals if v is not None)
        fixed = len(set(vals)) == 1
        print(f"{k:38s} {str(vals):34s} {edge}/{len(vals)}   {'SÍ' if fixed else 'no'}")
    for i, name in enumerate(rows[0].get("objectives", ["obj0", "obj1", "obj2"])):
        v = [r["best_values"][i] for r in rows]
        sd = st.pstdev(v) if len(v) > 1 else 0.0
        print(
            f"objetivo {name:10s}: media={st.mean(v):+.3f} sd={sd:.3f} min={min(v):+.3f} max={max(v):+.3f}"
        )
    print(
        "\nLectura: parámetros 'fijo=SÍ' = el sampler converge ahí; muchos en 'borde' = "
        "el espacio es demasiado estrecho; objetivo con sd alta = el ruido supera la señal."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--runner", choices=["docker", "local"], default=None)
    ap.add_argument(
        "--timeout", type=int, default=6 * 3600, help="segundos por semilla"
    )

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--runner", choices=["docker", "local"], default=None)
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="segundos por semilla")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.analyze_only:
        analyze(sorted(DIAG_DIR.glob("best_params_seed*.json")))
        return
    runner = detect_runner(a.runner)
    print(f"runner={runner}; semillas={a.seeds}; trials={a.trials}")
    backup = None
    if BEST_PARAMS.exists() and not a.dry_run:
        backup = BEST_PARAMS.with_suffix(".json.pre_script2")
        shutil.move(BEST_PARAMS, backup)  # si existe, ejecutar() omite el optimizador
        print(f"  (best_params.json respaldado en {backup.name})")
    done = []
    try:
        for s in a.seeds:
            print(f"\n── semilla {s} ──")
            if BEST_PARAMS.exists() and not a.dry_run:
                BEST_PARAMS.unlink()
            f = run_seed(s, a.trials, runner, a.timeout, a.dry_run)
            if f:
                done.append(f)
    finally:
        if backup and backup.exists():
            shutil.move(backup, BEST_PARAMS)
            print(f"\n(best_params.json original restaurado)")
    if not a.dry_run:
        analyze(done)


if __name__ == "__main__":
    main()
