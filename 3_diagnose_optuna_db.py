#!/usr/bin/env python3
"""
Script 3 — Diagnóstico de la BD de Optuna (SQLite) del optimizador REII.

Errores de la versión anterior, corregidos aquí:
  * Leía `trials.value`. El estudio es MULTIOBJETIVO (3 objetivos): esa columna es NULL
    y los valores viven en `trial_values(trial_id, objective, value)`.
  * Buscaba user attrs 'pass'/'stage'/'result_code' que el código nunca escribe
    (solo coverage_primary, agreement_sym, n_clusters, bottleneck_method, y solo en
    etapa 2). Resultado: siempre decía "sin centinelas".
  * El centinela real es _SENTINEL_TRIAL = (-2.0, -1.0, 0.0), no -999.
  * Ahora el espacio de búsqueda se lee de `trial_params.distribution_json` (la verdad
    del estudio), no de suposiciones.

Etapa 1 (study *_stage1_wc_*): objetivos = (ARI, coverage, -K)
Etapa 2 (study *_stage2_all_*): objetivos = (coverage, agreement_sym, -K)
Direcciones: maximize, maximize, minimize  (minimize sobre -K  ==> PREMIA más clases).

Uso:
  python 3_diagnose_optuna_db.py [--db data/reii_optuna.db] [--study NOMBRE]
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from diag_common import DATA_DIR, REPO, SENTINEL

DIRS = {
    0: "MINIMIZE?",
    1: "MINIMIZE",
    2: "MAXIMIZE",
}  # enum de Optuna: MINIMIZE=1, MAXIMIZE=2


def find_db(explicit):
    if explicit:
        return Path(explicit)
    cands = [Path(p) for p in glob.glob(str(DATA_DIR / "**" / "*.db"), recursive=True)]
    cands += [Path(p) for p in glob.glob(str(REPO / "*.db"))]
    for c in cands:
        try:
            con = sqlite3.connect(c)
            ok = con.execute(
                "SELECT name FROM sqlite_master WHERE name='trial_values'"
            ).fetchone()
            con.close()
            if ok:
                return c
        except sqlite3.Error:
            continue
    return None


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 4 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def dominates(a, b, dirs):
    better = False
    for x, y, d in zip(a, b, dirs):
        if d == 2:  # maximize
            if x < y:
                return False
            better |= x > y
        else:
            if x > y:
                return False
            better |= x < y
    return better


def diagnose_study(con, sid, name):
    print("\n" + "=" * 72 + f"\nESTUDIO {sid}: {name}\n" + "=" * 72)
    dirs = [
        d
        for (d,) in con.execute(
            "SELECT direction FROM study_directions WHERE study_id=? ORDER BY objective",
            (sid,),
        )
    ]
    dirs = [2 if str(d).upper().startswith("MAX") or d == 2 else 1 for d in dirs]
    stage = (
        "etapa 1 (wc_only): (ARI, coverage, -K)"
        if "stage1" in name
        else "etapa 2 (all): (coverage, agreement_sym, -K)"
        if "stage2" in name
        else "etapa desconocida"
    )
    print(f"Direcciones: {['MAX' if d == 2 else 'MIN' for d in dirs]} · {stage}")
    states = Counter(
        s for (s,) in con.execute("SELECT state FROM trials WHERE study_id=?", (sid,))
    )
    print(f"Trials por estado: {dict(states)}")

    vals = defaultdict(dict)
    for tid, obj, v, vt in con.execute(
        "SELECT tv.trial_id, tv.objective, tv.value, tv.value_type FROM trial_values tv "
        "JOIN trials t ON t.trial_id=tv.trial_id WHERE t.study_id=?",
        (sid,),
    ):
        vals[tid][obj] = float(v) if v is not None else float("nan")
    params = defaultdict(dict)
    dists = {}
    for tid, pn, pv, dj in con.execute(
        "SELECT p.trial_id, p.param_name, p.param_value, p.distribution_json FROM trial_params p "
        "JOIN trials t ON t.trial_id=p.trial_id WHERE t.study_id=?",
        (sid,),
    ):
        d = json.loads(dj)
        dists[pn] = d
        attrs = d.get("attributes", d)
        if "choices" in attrs:
            pv = attrs["choices"][int(pv)]
        params[tid][pn] = pv
    num = {tid: tuple(v[i] for i in sorted(v)) for tid, v in vals.items()}
    if not num:
        print("  (sin valores: ¿estudio vacío o aún corriendo?)")
        return
    n_obj = len(next(iter(num.values())))
    sent = [t for t, v in num.items() if tuple(v) == SENTINEL]
    valid = [t for t in num if t not in sent]
    print(
        f"Centinelas {SENTINEL}: {len(sent)}/{len(num)} ({100 * len(sent) / len(num):.0f}%) "
        f"— presupuesto desperdiciado en trials inviables"
    )

    # Causa de los centinelas inferible desde parámetros (gates baratos)
    inf_gate = 0
    for t in sent:
        p = params[t]
        if p.get("coref_context_units_tight", 0) >= p.get(
            "coref_context_units_loose", 99
        ) or p.get("hdbscan_uce_min_cluster_frac", 1) <= p.get(
            "hdbscan_uce_min_cluster_frac_loose", 0
        ):
            inf_gate += 1
    print(
        f"  de los cuales, infactibles por gates tight<loose / frac>frac_loose: {inf_gate} "
        f"(se evitan filtrando el espacio, no evaluando)"
    )

    if not valid:
        print("  Ningún trial válido.")
        return
    K = Counter(int(-num[t][-1]) for t in valid)
    print(f"Distribución de K (clases) en trials válidos: {dict(sorted(K.items()))}")
    for i in range(n_obj):
        col = [num[t][i] for t in valid]
        print(
            f"  objetivo[{i}]: min={min(col):+.3f} media={sum(col) / len(col):+.3f} max={max(col):+.3f}"
        )

    # Frente de Pareto propio
    front = [
        t
        for t in valid
        if not any(dominates(num[o], num[t], dirs) for o in valid if o != t)
    ]
    uniq = {tuple(round(x, 6) for x in num[t]) for t in front}
    print(
        f"Frente de Pareto: {len(front)} trials, {len(uniq)} puntos distintos en objetivo-espacio"
    )
    for pt in sorted(uniq, reverse=True):
        print(f"   {pt}")

    # Parámetros ganadores vs bordes (leídos de distribution_json)
    best = max(valid, key=lambda t: (num[t][0], num[t][1]))
    print(f"\nTrial #{best} (mejor 1er objetivo) — ¿en el borde del espacio?")
    for pn, v in sorted(params[best].items()):
        a = dists[pn].get("attributes", dists[pn])
        lo, hi = a.get("low"), a.get("high")
        flag = ""
        if lo is not None and (v == lo or v == hi):
            flag = "   ◄ BORDE"
        elif (
            "choices" in a and a["choices"] and v in (a["choices"][0], a["choices"][-1])
        ):
            flag = "   ◄ extremo de la lista"
        rng = f"[{lo}, {hi}]" if lo is not None else f"opciones={a.get('choices')}"
        print(f"   {pn:38s} {v!s:>8}   {rng}{flag}")

    # Plateaus y parámetros inertes
    by_vals = defaultdict(list)
    for t in valid:
        by_vals[tuple(round(x, 6) for x in num[t])].append(t)
    dup = {k: v for k, v in by_vals.items() if len(v) > 1}
    print(
        f"\nPuntos de objetivo repetidos por ≥2 trials con parámetros distintos: {len(dup)}"
    )
    print(
        "Correlación de Spearman parámetro↔objetivo (|ρ|<0.10 en todos = parámetro probablemente INERTE):"
    )
    for pn in sorted(dists):
        xs = [float(params[t][pn]) for t in valid if pn in params[t]]
        if len(set(xs)) < 2:
            print(f"   {pn:38s} sin variación")
            continue
        rhos = []
        for i in range(n_obj):
            ys = [num[t][i] for t in valid if pn in params[t]]
            r = spearman(xs, ys)
            rhos.append("  n/a " if r is None else f"{r:+.2f}")
        inert = all(
            r.strip() != "n/a" and abs(float(r)) < 0.10
            for r in rhos
            if r.strip() != "n/a"
        ) and any(r.strip() != "n/a" for r in rhos)
        print(f"   {pn:38s} ρ={rhos}{'   ◄ inerte' if inert else ''}")

    ua = Counter(
        k
        for (k,) in con.execute(
            "SELECT key FROM trial_user_attributes ua JOIN trials t ON t.trial_id=ua.trial_id WHERE t.study_id=?",
            (sid,),
        )
    )
    print(f"\nUser attrs presentes: {dict(ua) or 'ninguno (normal en etapa 1)'}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", default=None)
    ap.add_argument("--study", default=None, help="subcadena del nombre del estudio")
    a = ap.parse_args()
    db = find_db(a.db)
    if not db or not db.exists():
        print("✗ No encuentro una BD de Optuna. Pasa --db data/<archivo>.db")
        sys.exit(1)
    print(f"BD: {db}")
    con = sqlite3.connect(db)
    studies = con.execute(
        "SELECT study_id, study_name FROM studies ORDER BY study_id"
    ).fetchall()
    print(f"Estudios: {[s[1] for s in studies]}")
    for sid, name in studies:
        if a.study and a.study not in name:
            continue
        diagnose_study(con, sid, name)


if __name__ == "__main__":
    main()
