#!/usr/bin/env python3
"""
Script 3: Diagnostica la base de Optuna (reii_optuna.db).
Reporta: trials centinela, duplicados, Pareto, distribución de parámetros.

Uso:
  python 3_diagnose_optuna_db.py --db /ruta/a/reii/reii_optuna.db --out report.md

Comprueba mis hipótesis con datos reales:
  - ¿Hay trials que devolvieron centinela (inválidos)?
  - ¿Hay duplicados en warm-start (antes del fix)?
  - ¿Cuántos trials válidos en etapa 2 vs etapa 1?
"""

import json
import sqlite3
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path


def diagnose(db_path):
    """Lee la base de Optuna y reporta."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except sqlite3.DatabaseError as e:
        print(f"✗ Error al abrir la base: {e}")
        print(f"  ¿El archivo es una base SQLite válida?")
        return {}

    # Tabla: studies, trials, trial_params, trial_user_attributes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}

    if not tables:
        print("✗ Base de datos vacía (sin tablas)")
        return {}

    print(f"Tablas encontradas: {', '.join(sorted(tables))}\n")

    if "trials" not in tables:
        print("✗ No hay tabla 'trials' en la base (formato incorrecto?)")
        return {}

    # Leer trials
    try:
        cursor.execute(
            """
            SELECT id, state, value, datetime_start, datetime_complete
            FROM trials ORDER BY id
            """
        )
        trials = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"✗ Error al leer trials: {e}")
        return {}

    print(f"Total de trials: {len(trials)}")

    # Estados
    states = Counter(t[1] for t in trials)
    print(f"\nEstados de trials:")
    for state, count in sorted(states.items()):
        print(f"  {state}: {count}")

    # Trials completados (COMPLETE/FAIL/PRUNED)
    completed = [t for t in trials if t[1] in ("COMPLETE", "FAIL", "PRUNED")]
    print(f"\nTrials completados: {len(completed)}")

    # Valores
    values = [t[2] for t in completed if t[2] is not None]
    if values:
        print(f"Utilidad:")
        print(f"  Media: {sum(values) / len(values):.4f}")
        print(f"  Min: {min(values):.4f}")
        print(f"  Max: {max(values):.4f}")
        print(f"  # NaN/None: {len([v for v in values if v is None])}")

    # Detectar centinelas (trials devueltos sin completar la etapa 2)
    cursor.execute(
        """
        SELECT t.id, t.state, t.value, ua.key, ua.value
        FROM trials t
        LEFT JOIN trial_user_attributes ua ON t.id = ua.trial_id
        WHERE ua.key IN ('pass', 'stage', 'result_code')
        ORDER BY t.id
        """
    )
    trial_attrs = defaultdict(dict)
    for trial_id, state, value, key, val in cursor.fetchall():
        trial_attrs[trial_id][key] = val

    centinels = []
    for t in completed:
        trial_id = t[0]
        if trial_attrs[trial_id].get("pass") == "2":
            if t[2] is None or (
                isinstance(t[2], float) and abs(t[2] - (-999.0)) < 0.01
            ):
                centinels.append(trial_id)

    if centinels:
        print(f"\n⚠ Trials centinela (etapa 2 inválida): {len(centinels)}")
        print(f"  IDs: {centinels[:10]}{'...' if len(centinels) > 10 else ''}")
    else:
        print(f"\n✓ Sin trials centinela detectados")

    # Duplicados en warm-start
    cursor.execute(
        """
        SELECT tp.trial_id, tp.param_name, tp.param_value
        FROM trial_params tp
        ORDER BY tp.trial_id, tp.param_name
        """
    )
    trial_param_sets = defaultdict(dict)
    for trial_id, param_name, param_value in cursor.fetchall():
        try:
            trial_param_sets[trial_id][param_name] = json.loads(param_value)
        except:
            trial_param_sets[trial_id][param_name] = param_value

    param_hashes = {}
    for trial_id, params in trial_param_sets.items():
        h = json.dumps(params, sort_keys=True)
        if h not in param_hashes:
            param_hashes[h] = []
        param_hashes[h].append(trial_id)

    duplicates = {h: ids for h, ids in param_hashes.items() if len(ids) > 1}
    if duplicates:
        dup_count = sum(len(ids) - 1 for ids in duplicates.values())
        print(f"\n⚠ Parámetros duplicados: {dup_count} trials innecesarios")
        for h, ids in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)[
            :3
        ]:
            print(
                f"  Aparece {len(ids)}× en trials {ids[:3]}{'...' if len(ids) > 3 else ''}"
            )
    else:
        print(f"\n✓ Sin duplicados detectados")

    # Distribucion de parámetros en cada etapa
    cursor.execute(
        """
        SELECT tp.trial_id, tp.param_name, tp.param_value, ua.value as stage
        FROM trial_params tp
        JOIN trial_user_attributes ua ON tp.trial_id = ua.trial_id
        WHERE ua.key = 'stage'
        """
    )
    params_by_stage = defaultdict(lambda: defaultdict(list))
    for trial_id, param_name, param_value, stage in cursor.fetchall():
        try:
            val = json.loads(param_value)
        except:
            val = param_value
        params_by_stage[stage][param_name].append(val)

    print(f"\nDistribución de parámetros por etapa:")
    for stage in sorted(params_by_stage.keys()):
        print(f"  Etapa {stage}:")
        for param, vals in sorted(params_by_stage[stage].items()):
            if isinstance(vals[0], (int, float)):
                print(
                    f"    {param}: [{min(vals):.2f}, {max(vals):.2f}] (n={len(vals)})"
                )
            else:
                unique = set(str(v) for v in vals)
                print(f"    {param}: {len(unique)} valores únicos de {len(vals)}")

    # Frente de Pareto (si hay objetivos múltiples)
    cursor.execute(
        """
        SELECT id, value FROM trials
        WHERE state = 'COMPLETE' AND value IS NOT NULL
        ORDER BY value DESC
        """
    )
    valid_trials = list(cursor.fetchall())
    if valid_trials:
        best_value = max(t[1] for t in valid_trials)
        near_pareto = [t for t in valid_trials if t[1] >= best_value * 0.95]
        print(f"\nFrente de Pareto (últimos 5% del mejor):")
        print(f"  Mejor: {best_value:.4f}")
        print(f"  Trials en frente (>95%): {len(near_pareto)}")
        if near_pareto:
            print(f"  Trials: {[t[0] for t in near_pareto[:10]]}")

    conn.close()
    return trial_attrs


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Ruta a reii_optuna.db")
    parser.add_argument("--out", help="Guardar reporte en fichero (opcional)")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"✗ Base no existe: {db_path}")
        return

    print("=" * 70)
    print("SCRIPT 3: Diagnosticar base de Optuna")
    print("=" * 70)
    print(f"DB: {db_path}\n")

    diagnose(db_path)

    print("\n" + "=" * 70)
    print("INTERPRETACIÓN:")
    print("  Centinelas > 0: el problema de no persistir docs incompletos existe")
    print(
        "  Duplicados > 0: warm-start estaba tirando parámetros duplicados (bug fix P2)"
    )
    print("  Distribución: ¿explora bien o se queda en rincones?")


if __name__ == "__main__":
    main()
