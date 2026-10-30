#!/usr/bin/env python3
"""Script 2 - Estabilidad del optimizador entre semillas (Optuna / REII).

Pregunta: con distinto seed de TPE, ?el optimizador converge al mismo punto fijo
o explora? ?Los ganadores caen en los bordes del espacio?

Diseno (verificado contra src/reii/main_workflow.py):
  1. check_source() aborta (codigo 2) si main_workflow.py no contiene los flags
     que este script usa. Antes, un flag inexistente se ignoraba en silencio y
     el script se convertia en un no-op que "pasaba" sin probar nada.
  2. Nombre de estudio UNICO por semilla y corrida:
     f"{prefix}_seed{seed}_{YYYYmmdd_HHMMSS}". Nunca se reutiliza un nombre:
     main_workflow.py usa load_if_exists=True y sella el estudio con el
     fingerprint del corpus, asi que reutilizar el nombre contamina el estudio.
  3. Flujo util: mueve best_params.json a .pre_script2, lanza por semilla
     `--force --optimize-only --trials N --seed S --study-name NAME` y lo
     restaura en finally. --optimize-only se detiene tras el optimizador.
  4. NO depende de la linea "[OPTIMIZER-RESULT]" (no existe en el codigo).
     Verifica el resultado en el sqlite (data/reii_optuna.db): estudios cuyo
     nombre empieza por NAME, numero de trials y centinelas (-2,-1,0) leidos de
     `trial_values` (NO de `trials.value`, que es NULL en multiobjetivo).
     Imprime validez % por etapa autodescubriendo el sufijo del nombre
     (_stage1_wc / _stage2_select).
  5. Copia best_params.json a data/diag/best_params_seed*.json y analiza los
     params realmente escritos. El tercer valor es ``-K_util`` (clases con
     peso suficiente), no el K bruto: ``K_util <= K real``, asi que
     `-best_values[2]` SUBESTIMA el K de la particion cuando el metodo dejo
     clases por debajo del piso. El nombre exacto esta en ``objectives``.
     Ver docs/design_v5/AHORRO_K_Y_PERSISTENCIA.md §3.
     Avisa si sd=0 en todas las semillas (estudio reutilizado / no ejecutado).

Uso:
  python -m reii.diag.cli trials --seeds 0 1 2 --trials 300
  python -m reii.diag.cli trials --seeds 0 1 2 --trials 300 --stage1-fraction 0.2 --no-warm-start
  python -m reii.diag.cli trials --analyze-only
  python -m reii.diag.cli trials --dry-run --seeds 0 --trials 20
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import statistics as st
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reii.diag.common import (
    BEST_PARAMS,
    DATA_DIR,
    DIAG_DIR,
    REPO,
    SENTINEL,
    at_boundary,
    detect_runner,
    run_to_log,
    workflow_cmd,
)

OPTUNA_DB = DATA_DIR / "reii_optuna.db"
MAIN_WORKFLOW = REPO / "src" / "reii" / "main_workflow.py"

# Tokens que deben existir en main_workflow.py para que este script pruebe algo.
REQUIRED_SOURCE_TOKENS = (
    "_cli_overrides",
    "--optimize-only",
    "optimize_stage1_fraction",
    "no-optuna-warm-start",
)

# Tolerancia al comparar valores de trial con el centinela.
SENTINEL_TOL = 1e-6


def _utf8_stdout() -> None:
    """Salida UTF-8 (evita UnicodeEncodeError en consolas cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def check_source() -> None:
    """Aborta con codigo 2 si main_workflow.py no soporta los flags usados."""
    if not MAIN_WORKFLOW.exists():
        print(f"[ERROR] no existe {MAIN_WORKFLOW}")
        sys.exit(2)
    text = MAIN_WORKFLOW.read_text(encoding="utf-8", errors="replace")
    missing = [tok for tok in REQUIRED_SOURCE_TOKENS if tok not in text]
    if missing:
        print("[ERROR] main_workflow.py NO soporta los flags que este script usa.")
        print(f"        faltan: {missing}")
        print("        Abortando: sin esos flags el script seria un no-op silencioso.")
        sys.exit(2)
    print(f"[source] OK: {MAIN_WORKFLOW.name} contiene {list(REQUIRED_SOURCE_TOKENS)}")


# ── Lectura del sqlite de Optuna ────────────────────────────────────────────


def _study_rows(prefix: str) -> Optional[List[Tuple[str, List[List[float]]]]]:
    """[(study_name, [values por trial])] de estudios cuyo nombre empieza por prefix.

    Devuelve None si no existe el sqlite. Los valores se leen de `trial_values`
    (trials.value es NULL en estudios multiobjetivo).
    """
    if not OPTUNA_DB.exists():
        return None
    uri = f"file:{OPTUNA_DB.as_posix()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        print(f"  [!] no se pudo abrir {OPTUNA_DB}: {exc}")
        return None
    try:
        con.row_factory = sqlite3.Row
        studies = con.execute(
            "SELECT study_id, study_name FROM studies WHERE study_name LIKE ?",
            (prefix + "%",),
        ).fetchall()
        out: List[Tuple[str, List[List[float]]]] = []
        for study in studies:
            rows = con.execute(
                "SELECT t.trial_id, tv.objective, tv.value "
                "FROM trials t LEFT JOIN trial_values tv ON tv.trial_id = t.trial_id "
                "WHERE t.study_id = ? ORDER BY t.trial_id, tv.objective",
                (study["study_id"],),
            ).fetchall()
            per_trial: Dict[int, List[float]] = {}
            for row in rows:
                values = per_trial.setdefault(int(row["trial_id"]), [])
                if row["value"] is not None:
                    values.append(float(row["value"]))
            out.append((str(study["study_name"]), list(per_trial.values())))
        return out
    finally:
        con.close()


def _is_sentinel(values: List[float]) -> bool:
    if len(values) != len(SENTINEL):
        return False
    return all(abs(v - s) <= SENTINEL_TOL for v, s in zip(values, SENTINEL))


def _stage_of(study_name: str, prefix: str) -> str:
    """Etapa segun el sufijo que anade Optimizador._run_single_stage."""
    tail = study_name[len(prefix) :] if study_name.startswith(prefix) else study_name
    if "_stage1_wc" in tail:
        return "stage1_wc"
    if "_stage2_select" in tail:
        return "stage2_select"
    if "_stage2_all" in tail:  # nombre historico (antes de select)
        return "stage2_all"
    return "single_stage"


def verify_run(prefix: str, n_trials: int) -> Optional[Dict[str, Any]]:
    """Verifica en el sqlite que la corrida se ejecuto y cuenta centinelas."""
    n_trials = int(n_trials)
    rows = _study_rows(prefix)
    if rows is None:
        print(f"  [!] no existe {OPTUNA_DB}: no se puede verificar la corrida")
        return None
    if not rows:
        print(f"  [!] ningun estudio en {OPTUNA_DB.name} empieza por {prefix!r}")
        return None

    total = 0
    per_stage: Dict[str, List[int]] = {}  # etapa -> [trials, centinelas]
    for name, trials in rows:
        stage = _stage_of(name, prefix)
        n = len(trials)
        n_sent = sum(1 for values in trials if _is_sentinel(values))
        total += n
        acc = per_stage.setdefault(stage, [0, 0])
        acc[0] += n
        acc[1] += n_sent
        valid_pct = 100.0 * (n - n_sent) / n if n else 0.0
        print(
            f"  estudio {name}: {n} trials, {n_sent} centinela, {valid_pct:.1f}% validos"
        )

    for stage in sorted(per_stage):
        n, n_sent = per_stage[stage]
        valid_pct = 100.0 * (n - n_sent) / n if n else 0.0
        print(f"  etapa {stage:12s}: {n - n_sent}/{n} validos ({valid_pct:.1f}%)")

    if total < n_trials:
        print(
            f"  [!] trials acumulados={total} < n_trials={n_trials}: "
            "la corrida no consumio el presupuesto (revisar el log)"
        )
    return {"n_trials": total, "per_stage": per_stage}


# ── Ejecucion por semilla ───────────────────────────────────────────────────


def run_seed(
    seed: int,
    args: argparse.Namespace,
    runner: str,
    stamp: str,
) -> Tuple[Optional[Path], str]:
    """Lanza una semilla, verifica el sqlite y copia best_params.json si existe."""
    name = f"{args.study_prefix}_seed{seed}_{stamp}"
    flags = [
        "--force",
        "--optimize-only",
        "--trials",
        str(args.trials),
        "--seed",
        str(seed),
        "--study-name",
        name,
    ]
    if args.stage1_fraction is not None:
        flags += ["--stage1-fraction", str(args.stage1_fraction)]
    if args.no_warm_start:
        flags += ["--no-optuna-warm-start"]

    print(f"\n-- semilla {seed} | estudio {name} --")
    log_path = DIAG_DIR / f"seed{seed}_{stamp}.log"
    run_to_log(workflow_cmd(runner, *flags), log_path, args.timeout, args.dry_run)
    if args.dry_run:
        print("  [dry-run] sin ejecucion ni verificacion")
        return None, name

    verify_run(name, args.trials)

    if not BEST_PARAMS.exists():
        print(f"  [!] no se genero {BEST_PARAMS.name} (sin trials validos)")
        return None, name
    dest = DIAG_DIR / f"best_params_seed{seed}.json"
    shutil.copy(BEST_PARAMS, dest)
    print(f"  {BEST_PARAMS.name} -> {dest}")
    return dest, name


# ── Analisis ────────────────────────────────────────────────────────────────


def analyze(files: List[Path]) -> None:
    rows: List[Dict[str, Any]] = []
    for path in files:
        try:
            rows.append(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [!] no se pudo leer {path}: {exc}")
    if not rows:
        print("Sin resultados que analizar.")
        return

    print("\n" + "=" * 72)
    print(f"ANALISIS ({len(rows)} semillas)")
    print("=" * 72)

    params = sorted({k for r in rows for k in (r.get("params") or {})})
    print(f"{'parametro':38s} {'valores por semilla':34s} borde  fijo?")
    for key in params:
        values = [(r.get("params") or {}).get(key) for r in rows]
        edge = sum(1 for v in values if v is not None and at_boundary(key, v))
        fixed = len({json.dumps(v, sort_keys=True) for v in values}) == 1
        print(
            f"{key:38s} {str(values):34s} {edge}/{len(values)}   {'SI' if fixed else 'no'}"
        )

    names = rows[0].get("objectives") or ["obj0", "obj1", "obj2"]
    sds: List[float] = []
    for idx, obj_name in enumerate(names):
        values = [
            float(r["best_values"][idx])
            for r in rows
            if len(r.get("best_values") or []) > idx
        ]
        if not values:
            continue
        sd = st.stdev(values) if len(values) > 1 else 0.0
        sds.append(sd)
        print(
            f"objetivo {obj_name:16s}: media={st.mean(values):+.4f} sd={sd:.4f} "
            f"min={min(values):+.4f} max={max(values):+.4f}"
        )
        if idx == 2:  # neg_k_util (rama "all" legacy: neg_k)
            ks = [-v for v in values]
            print(
                f"K_util = -best_values[2]: media={st.mean(ks):.2f} "
                f"min={min(ks):.2f} max={max(ks):.2f}"
            )
            print(
                "    (K_util <= K real: las clases bajo el piso de masa no "
                "cuentan; el nombre exacto esta en 'objectives')"
            )

    if len(rows) >= 3 and sds and all(sd == 0.0 for sd in sds):
        print(
            "\n[!] sd=0 => sospecha de estudio reutilizado o de que el "
            "optimizador no se ejecuto"
        )

    print(
        "\nLectura: 'fijo=SI' = el sampler converge ahi; muchos en 'borde' = el "
        "espacio es demasiado estrecho; sd alta = el ruido supera la senal."
    )


# ── main ────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="reii.diag.cli trials",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument(
        "--stage1-fraction",
        type=float,
        default=None,
        help="fraccion de trials para la etapa 1 (default: el del Config)",
    )
    ap.add_argument(
        "--no-warm-start",
        action="store_true",
        help="propaga --no-optuna-warm-start (etapa 2 sin warm start)",
    )
    ap.add_argument("--runner", choices=["docker", "local"], default=None)
    ap.add_argument(
        "--timeout", type=int, default=6 * 3600, help="segundos por semilla"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--study-prefix",
        default="reii",
        help="prefijo del nombre de estudio (default: reii)",
    )
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args(argv)

    _utf8_stdout()

    if args.analyze_only:
        analyze(sorted(DIAG_DIR.glob("best_params_seed*.json")))
        return 0

    check_source()

    runner = detect_runner(args.runner)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(
        f"[config] runner={runner} seeds={args.seeds} trials={args.trials} "
        f"stage1_fraction={args.stage1_fraction} no_warm_start={args.no_warm_start} "
        f"stamp={stamp}"
    )

    if not args.dry_run:
        DIAG_DIR.mkdir(parents=True, exist_ok=True)

    backup: Optional[Path] = None
    if BEST_PARAMS.exists() and not args.dry_run:
        backup = BEST_PARAMS.with_suffix(".json.pre_script2")
        shutil.move(BEST_PARAMS, backup)  # si existe, ejecutar() omite el optimizador
        print(f"(respaldado {BEST_PARAMS.name} -> {backup.name})")

    done: List[Path] = []
    try:
        for seed in args.seeds:
            if BEST_PARAMS.exists() and not args.dry_run:
                BEST_PARAMS.unlink()
            dest, _name = run_seed(seed, args, runner, stamp)
            if dest is not None:
                done.append(dest)
    finally:
        if backup is not None and backup.exists():
            shutil.move(backup, BEST_PARAMS)
            print(f"\n(restaurado {BEST_PARAMS.name})")

    if not args.dry_run:
        analyze(done)

    return 0


def run(argv=None) -> int:
    """Contrato del CLI: lo invoca ``reii.diag.cli`` con el argv restante."""
    return int(main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(run())
