#!/usr/bin/env python
"""Simulador del espacio de busqueda v4 (actual) vs v5 (condicional).

Objetivo
--------
Medir, sin lanzar el optimizador ni importar `reii.main_workflow`, cuatro
cosas del diseno de `docs/design_v5/V5_SPACE.md`:

  1. Tasa de combinaciones que el gate estructural rechaza
     (`tight >= loose`, `frac <= frac_loose`) — los "centinelas de
     viabilidad" del probe v4.
  2. Cardinalidad del espacio (combinaciones unicas alcanzables) y su
     equivalente en bits (= dimensionalidad efectiva).
  3. Marginales por parametro, para detectar espacios que colapsan en los
     extremos (bordes clavados).
  4. Efecto de congelar parametros inertes (rho ~ 0) sobre la
     dimensionalidad efectiva.
  5. Presupuesto: cuantos draws hacen falta para que cada nivel marginal
     se visite (y para que TPE tenga muestra por grupo).

El muestreo se replica con las mismas semantica que usaria Optuna 3.6.2:
categorico = uniforme sobre la lista, int = uniforme en [lo, hi] inclusive,
float = uniforme en [lo, hi]. El espacio v5 se dibuja con la MISMA logica
que `Optimizador._suggest` + `_suggest_conditional` del diff (mapeo
monotono, sin distribuciones dinamicas categoricas: Optuna 3.6.2 lanza
`ValueError('CategoricalDistribution does not support dynamic value space')`,
medido).

Uso
---
    python scripts/playground/v5_space_sim.py            # 5000 draws
    python scripts/playground/v5_space_sim.py -n 20000
    python scripts/playground/v5_space_sim.py --no-source-check

Salida solo con los prefijos `->`, `[ok]`, `[!]`, `[X]`, `=>`.
No escribe archivos, no toca Docker, no importa el pipeline.
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import random
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Constantes del probe v4 (medidas)
# ─────────────────────────────────────────────────────────────────────────────
N_EMB = 528  # UCEs embeddables del corpus del probe (comentario del repo)
FLOOR = 5  # hdbscan_uce_min_cluster_floor
PROBE_TRIALS = 40  # etapa 1 del probe v4
PROBE_STARTUP = 25  # optimize_n_startup_trials vigente
PROBE_CENTINELS = 20
PROBE_GATE_CENTINELS = 13

MAIN_WORKFLOW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "reii",
    "main_workflow.py",
)

# ─────────────────────────────────────────────────────────────────────────────
# Espacios
# ─────────────────────────────────────────────────────────────────────────────
# v4: lo que hay hoy en Optimizador.get_search_space() (L8408-8432).
# min_forms_uc=[13,17] -> min_forms_uc_1 in (9,17); tsj=3 -> (2,5);
# n_total=528 -> k_hi = min(20, 52) = 20 -> cdh_target_k in (3,20).
OLD_SPACE = {
    "min_forms_uc_1": (9, 17),
    "forms_gap": (2, 7),
    "tsj": (2, 5),
    "cdh_target_k": (3, 20),
    "swap_iterations": [1, 2, 5],
    "min_r2_threshold": [0.03, 0.05, 0.08, 0.10],
    "similarity_threshold": [0.25, 0.35, 0.45, 0.55, 0.65],
    "coref_weight": [0.0, 0.1, 0.2, 0.3, 0.4],
    "uc_window_size": [1, 2, 3, 4, 5],
    "coref_context_units_tight": [1, 2, 3],
    "coref_context_units_loose": [3, 4, 5],
    "hdbscan_uce_min_cluster_frac": [0.01, 0.015, 0.02, 0.03, 0.04],
    "hdbscan_uce_min_cluster_frac_loose": [0.005, 0.01, 0.015, 0.02, 0.03],
}

FRAC_EPS = max(0.0005, round(1.0 / N_EMB, 5))
LATTICE = 0.0025  # cuantizacion para contar valores continuos de frac

# v5: condicional por construccion + bordes extendidos (V5_SPACE.md §1-2).
NEW_SPACE = {
    "min_forms_uc_1": (9, 17),
    "forms_gap": (2, 7),
    "tsj": (2, 5),
    "cdh_target_k": (3, 20),
    "swap_iterations": [0, 1, 2, 3, 5],
    "min_r2_threshold": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
    "similarity_threshold": [0.25, 0.35, 0.45, 0.55, 0.65],
    "coref_weight": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "uc_window_size": [1, 2, 3, 4, 5],
    # driver primero, dependiente despues (el orden no importa: _suggest hace
    # dos pasadas, pero se declara en este orden por legibilidad)
    "coref_context_units_loose": [3, 4, 5, 6, 8],
    "coref_context_units_tight": {
        "kind": "lt",
        "other": "coref_context_units_loose",
        "lo": 1,
    },
    "hdbscan_uce_min_cluster_frac_loose": [0.005, 0.01, 0.015, 0.02, 0.03],
    "hdbscan_uce_min_cluster_frac": {
        "kind": "gt",
        "other": "hdbscan_uce_min_cluster_frac_loose",
        "hi": 0.04,
        "eps": FRAC_EPS,
    },
}

# Congelados recomendados (rho ~ 0 en los 3 objetivos, n=11 -> |rho|<0.3 es ruido)
FREEZE = {"uc_window_size": 3, "min_forms_uc_1": 13}

BAR = "-" * 78


def _say(prefix: str, text: str = "") -> None:
    print(f"{prefix} {text}" if text else prefix)


# ─────────────────────────────────────────────────────────────────────────────
# Muestreo (espejo de Optimizador._suggest / _suggest_conditional)
# ─────────────────────────────────────────────────────────────────────────────
def draw(space, rng):
    """Un draw con la semantica de Optuna; devuelve el dict de params."""
    params = {}
    pending = {}
    for name, spec in space.items():
        if isinstance(spec, dict):
            pending[name] = spec
            continue
        if isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = spec
            if isinstance(lo, int) and isinstance(hi, int):
                params[name] = rng.randint(lo, hi)
            else:
                params[name] = rng.uniform(float(lo), float(hi))
        elif isinstance(spec, list):
            params[name] = rng.choice(spec)
        else:
            params[name] = spec

    for name, spec in pending.items():
        base = params[spec["other"]]
        if spec["kind"] == "lt":
            lo = int(spec.get("lo", 1))
            hi = int(base) - 1
            params[name] = rng.randint(lo, hi) if hi >= lo else lo
        else:  # "gt"
            lo = float(base) + float(spec["eps"])
            hi = float(spec["hi"])
            params[name] = rng.uniform(lo, hi) if hi >= lo else lo
    return params


def structural_violation(p):
    """Los dos gates duros de _evaluate_trial (L8560-8563)."""
    if p.get("coref_context_units_tight", 0) >= p.get("coref_context_units_loose", 99):
        return "tight>=loose"
    if p.get("hdbscan_uce_min_cluster_frac", 1.0) <= p.get(
        "hdbscan_uce_min_cluster_frac_loose", 0.0
    ):
        return "frac<=frac_loose"
    return None


def resolve_mcs(frac, n=N_EMB):
    """_resolve_hdbscan_uce_params: max(floor, round(frac * n_emb))."""
    return max(FLOOR, int(round(float(frac) * n)))


# ─────────────────────────────────────────────────────────────────────────────
# Cardinalidad
# ─────────────────────────────────────────────────────────────────────────────
def _spec_levels(spec):
    if isinstance(spec, tuple):
        lo, hi = spec
        return hi - lo + 1 if isinstance(lo, int) else 2
    if isinstance(spec, list):
        return len(spec)
    return 1


# Los dos pares acoplados: (dependiente, driver, tipo de gate).
PAIRS = (
    ("coref_context_units_tight", "coref_context_units_loose", "lt"),
    (
        "hdbscan_uce_min_cluster_frac",
        "hdbscan_uce_min_cluster_frac_loose",
        "gt",
    ),
)


def _pair_combos(space, dep, driver, kind):
    """Combos factibles (driver, dep) del par.

    Devuelve (por_valor, por_mcs): cuantos pares distintos se alcanzan si el
    dependiente toma valores discretos (grilla v4, o malla 0.0025 si es
    continuo) y cuantos min_cluster_size distintos resuelve (efecto real).
    """
    spec = space[dep]
    levels = space[driver]
    if isinstance(levels, dict) or isinstance(spec, dict) and spec["other"] != driver:
        return 0, 0

    if kind == "lt":
        if isinstance(spec, dict):
            dep_values = None  # derivado: [lo, driver-1] por construccion
        else:
            dep_values = [int(v) for v in spec]
        by_value = 0
        for lv in levels:
            lo = int(spec.get("lo", 1)) if isinstance(spec, dict) else -(10**9)
            hi = int(lv) - 1
            if dep_values is None:
                by_value += max(0, hi - lo + 1)
            else:
                by_value += sum(1 for v in dep_values if lo <= v <= hi)
        return by_value, by_value

    # kind == "gt" (fraccion tight > fraccion loose)
    eps = float(spec["eps"]) if isinstance(spec, dict) else 0.0
    hi = float(spec["hi"]) if isinstance(spec, dict) else max(spec)
    if isinstance(spec, dict):
        dep_values = None  # continuo: se enumera en la malla de 0.0025
    else:
        dep_values = [float(v) for v in spec]

    by_value = 0
    by_mcs = 0
    for lv in levels:
        lo = float(lv) + eps
        if hi < lo:
            continue
        if dep_values is None:
            first = math.ceil(lo / LATTICE - 1e-9) * LATTICE
            grid = []
            v = first
            while v <= hi + 1e-12:
                grid.append(round(v, 6))
                v += LATTICE
        else:
            grid = [v for v in dep_values if v > float(lv)]
        by_value += len(grid)
        mcs_l = resolve_mcs(lv)
        by_mcs += len({resolve_mcs(g) for g in grid if resolve_mcs(g) > mcs_l})
    return by_value, by_mcs


def cardinality(space):
    """Cardinalidad = prod(niveles escalares) x prod(combos factibles de cada par).

    Se excluyen del producto los miembros de los pares para no contar dos
    veces y para comparar v4 (miembros independientes) con v5 (derivados)
    bajo la misma vara. ``total_all`` es el producto sin filtrar el gate (solo
    tiene sentido cuando los miembros del par son independientes, es decir v4).
    """
    pair_members = set()
    for dep, driver, _ in PAIRS:
        pair_members.update({dep, driver})

    detail = {}
    total = 1
    for name, spec in space.items():
        if name in pair_members or isinstance(spec, dict):
            continue
        detail[name] = _spec_levels(spec)
        total *= detail[name]

    pairs = {}
    total_all = total
    for dep, driver, kind in PAIRS:
        by_value, by_mcs = _pair_combos(space, dep, driver, kind)
        pairs[dep] = (by_value, by_mcs)
        total *= by_value
        dep_spec = space[dep]
        if not isinstance(dep_spec, dict):
            total_all *= _spec_levels(space[driver]) * _spec_levels(dep_spec)
        else:
            total_all *= by_value  # condicional: no hay combos invalidos

    total_mcs = total
    for dep, (by_value, by_mcs) in pairs.items():
        if by_value:
            total_mcs = total_mcs / by_value * by_mcs
    return {
        "total": total,
        "total_all": total_all,
        "total_mcs": total_mcs,
        "detail": detail,
        "pairs": pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reportes
# ─────────────────────────────────────────────────────────────────────────────
def section_infeasible(n_draws, rng, old, new):
    _say("->", f"1. Tasa de inviabilidad estructural ({n_draws} draws uniformes)")
    rows = []
    for tag, space in (("v4", old), ("v5", new)):
        bad = collections.Counter()
        degen = 0
        feasible = 0
        for _ in range(n_draws):
            p = draw(space, rng)
            why = structural_violation(p)
            if why:
                bad[why] += 1
                continue
            feasible += 1
            # degeneracion semantica: los dos pases de HDBSCAN resuelven al
            # mismo min_cluster_size (gate numerico OK, efecto nulo)
            if resolve_mcs(p["hdbscan_uce_min_cluster_frac"]) == resolve_mcs(
                p["hdbscan_uce_min_cluster_frac_loose"]
            ):
                degen += 1
        rows.append((tag, bad, degen, feasible))

    for tag, bad, degen, feasible in rows:
        rate = sum(bad.values()) / n_draws
        _say(
            "  ",
            f"{tag}: inviables {rate:6.2%}  "
            f"(tight>=loose {bad['tight>=loose'] / n_draws:5.2%} · "
            f"frac<=frac_loose {bad['frac<=frac_loose'] / n_draws:5.2%})"
            f"   degenerados {degen / max(1, feasible):5.2%}",
        )

    old_rate = sum(rows[0][1].values()) / n_draws
    new_rate = sum(rows[1][1].values()) / n_draws
    if new_rate == 0.0:
        _say("[ok]", "el espacio v5 nunca genera una combinacion inviable por gate")
    else:
        _say("[X]", f"el espacio v5 genera {new_rate:.2%} de inviables (deberia ser 0)")

    # Contraste con el probe v4: 13/40 centinelas de gate.
    mix = PROBE_STARTUP / PROBE_TRIALS
    expected = old_rate * mix * PROBE_TRIALS
    _say(
        "=>",
        f"probe v4: {PROBE_GATE_CENTINELS}/{PROBE_TRIALS} centinelas de gate "
        f"(medido) vs {expected:.1f} esperados con {PROBE_STARTUP}/{PROBE_TRIALS} "
        f"draws uniformes ({mix:.0%} del presupuesto; el resto lo propone TPE)",
    )
    _say(
        "=>",
        f"en v5 esos {PROBE_GATE_CENTINELS} desaparecen; los "
        f"{PROBE_CENTINELS - PROBE_GATE_CENTINELS} restantes del probe son de "
        f"cobertura/K (otro problema: el rediseno no los toca)",
    )
    return old_rate, new_rate


def section_cardinality(old, new):
    _say("->", "2. Cardinalidad alcanzable (solo combinaciones factibles)")
    o = cardinality(old)
    n = cardinality(new)

    _say(
        "  ",
        f"v4: {o['total']:.3e} combos · {math.log2(o['total']):.1f} bits "
        f"· pares coref {o['pairs']['coref_context_units_tight'][0]} "
        f"· pares frac {o['pairs']['hdbscan_uce_min_cluster_frac'][0]}",
    )
    _say(
        "  ",
        f"v5: {n['total']:.3e} combos · {math.log2(n['total']):.1f} bits "
        f"· pares coref {n['pairs']['coref_context_units_tight'][0]} "
        f"· pares frac {n['pairs']['hdbscan_uce_min_cluster_frac'][0]} "
        f"(malla 0.0025)",
    )
    _say(
        "  ",
        f"resueltos a min_cluster_size (n={N_EMB}): v4 "
        f"{o['total_mcs']:.3e} · v5 {n['total_mcs']:.3e} combos",
    )
    ratio = n["total"] / o["total"]
    _say(
        "=>",
        f"espacio nominal x{ratio:.1f} · todo el espacio v5 es factible "
        f"(en v4 solo {o['total'] / o['total_all']:.1%} de sus combos "
        f"pasaba el gate)",
    )
    if ratio < 1.0:
        _say("[X]", "el espacio v5 se degrado (cardinalidad menor que v4)")
    else:
        _say("[ok]", "el espacio v5 no degrada la cobertura (cardinalidad >= v4)")

    for k, a in sorted(o["detail"].items()):
        b = n["detail"].get(k, a)
        if a != b:
            _say("  ", f"[borde] {k}: {a} -> {b} niveles (x{b / a:.2f})")
    for dep, (bv, bm) in sorted(n["pairs"].items()):
        ov = o["pairs"][dep][0]
        _say("  ", f"[borde] {dep}: {ov} -> {bv} combos (mcs: {bm})")
    return o["total"], n["total"]


def section_marginals(n_draws, rng, space, tag, frozen=()):
    _say("->", f"3. Marginales ({tag}) — masa en los extremos")
    counts = collections.defaultdict(collections.Counter)
    for _ in range(n_draws):
        for k, v in draw(space, rng).items():
            if k in frozen:
                continue
            if isinstance(v, float):
                counts[k][round(v, 5)] += 1
            else:
                counts[k][v] += 1
    alarm = False
    for k in space:
        if k in frozen or isinstance(space[k], dict):
            continue
        c = counts[k]
        vals = sorted(c)
        if not vals:
            continue
        extreme = (c[vals[0]] + c[vals[-1]]) / n_draws
        # la masa uniforme esperada en los dos extremos es 2/L: el aviso es
        # relativo a esa linea base (si no, todo parametro de 4 niveles alarma)
        baseline = 2.0 / max(1, _spec_levels(space[k]))
        ratio = extreme / baseline if baseline else 0.0
        top = max(c.items(), key=lambda kv: kv[1])
        flag = "  <-- sobre-representado" if ratio > 1.5 else ""
        alarm = alarm or ratio > 1.5
        _say(
            "  ",
            f"{k:38s} niveles={len(c):3d}  extremo={extreme:5.1%} "
            f"(x{ratio:.2f} uniforme)  moda={top[0]} "
            f"({top[1] / n_draws:4.1%}){flag}",
        )
    if alarm:
        _say("[!]", "algun marginal sobre-representa los bordes (>1.5x uniforme)")
    else:
        _say("[ok]", "ningun marginal colapsa en los extremos")
    return alarm


def section_freeze(old, new):
    _say("->", "4. Efecto de congelar parametros inertes")
    frozen_total = 1
    for k in FREEZE:
        frozen_total *= _spec_levels(new[k])
    n_total = cardinality(new)["total"]
    o_total = cardinality(old)["total"]
    f_total = n_total / frozen_total
    _say(
        "  ",
        f"v5 completo  : {math.log2(n_total):5.1f} bits "
        f"(13 parametros, 2 condicionales)",
    )
    _say(
        "  ",
        f"v5 congelado : {math.log2(f_total):5.1f} bits "
        f"(11 parametros: -{math.log2(frozen_total):.1f} bits por "
        f"{', '.join(f'{k}={v}' for k, v in FREEZE.items())})",
    )
    _say("  ", f"v4 (referencia): {math.log2(o_total):5.1f} bits")
    if math.log2(f_total) <= math.log2(o_total):
        _say(
            "[ok]",
            "con los 2 congelados el espacio queda igual o mas chico que v4, "
            "pero 100% factible",
        )
    else:
        _say(
            "[!]",
            f"con los 2 congelados el espacio aun supera v4 en "
            f"{math.log2(f_total) - math.log2(o_total):.1f} bits",
        )
    # marginales del congelado: no hay nada que medir, se informa el valor
    for k, v in FREEZE.items():
        _say("  ", f"[congelado] {k} = {v} (constante en el dict del espacio)")
    return math.log2(f_total)


def section_budget(new):
    _say("->", "5. Presupuesto: niveles marginales sin visitar")
    header = (
        "  " + "parametro".ljust(30) + "".join(f"n={n:<5d}" for n in (40, 60, 80, 120))
    )
    _say("  ", header.strip())
    worst = []
    for name, spec in new.items():
        if isinstance(spec, dict):
            continue
        levels = _spec_levels(spec)
        if levels <= 1:
            continue
        row = f"  {name[:30]:30s}"
        missing = []
        for n in (40, 60, 80, 120):
            exp_missing = levels * (1.0 - 1.0 / levels) ** n
            missing.append(exp_missing)
            row += f"{exp_missing:6.2f}"
        worst.append((missing[-1], name, levels))
        if missing[-1] > 0.50:
            row += "  <-- exige mas presupuesto"
        _say("  ", row.strip())

    # n necesario para cubrir todos los niveles con 95% de confianza
    _say("  ", "")
    _say("  ", "n para P(todos los niveles vistos) >= 95%:")
    for name, spec in new.items():
        if isinstance(spec, dict):
            continue
        levels = _spec_levels(spec)
        if levels <= 1:
            continue
        n95 = math.ceil(math.log(0.05 / levels) / math.log(1.0 - 1.0 / levels))
        _say("  ", f"  {name[:34]:34s} niveles={levels:2d}  n>={n95}")

    # TPE: n_below = min(ceil(0.1*n_despues_startup), 25) y Parzen necesita
    # >= ~10 puntos por grupo y por dimension.
    _say("  ", "")
    for startup, total in ((25, 40), (12, 120), (12, 200)):
        after = max(0, total - startup)
        n_below = min(math.ceil(0.1 * after), 25)
        _say(
            "  ",
            f"n_startup={startup:2d} + {after:3d} TPE -> n_below={n_below:2d} "
            f"{'(Parzen degenerado)' if n_below < 10 else '(ok)'}",
        )
    _say(
        "=>",
        "n_startup=12 y --trials>=120 dejan n_below>=10; con 40 trials y "
        "startup=25 TPE nunca llega a modelar",
    )


def section_source_check():
    _say("->", "0. Chequeo del diff en main_workflow.py (solo lectura/AST)")
    if not os.path.exists(MAIN_WORKFLOW):
        _say("[!]", f"no existe {MAIN_WORKFLOW}; se simula la referencia")
        return False
    with open(MAIN_WORKFLOW, "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    markers = {
        "espacio condicional": "def _suggest_conditional",
        "guard": "def _make_guarded_objective",
        "resumen": "def _sentinel_breakdown",
    }
    missing = [k for k, v in markers.items() if v not in src]
    if missing:
        _say(
            "[!]",
            "diff aun NO aplicado (se simula la implementacion de referencia): "
            + ", ".join(missing),
        )
        return False
    _say("[ok]", "main_workflow.py ya contiene el espacio condicional y el guard")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20251028)
    ap.add_argument("--no-source-check", action="store_true")
    args = ap.parse_args(argv)

    if args.draws <= 0:
        _say("[X]", "n de draws debe ser > 0")
        return 2

    print(BAR)
    _say("->", f"V5 space sim · {args.draws} draws · seed={args.seed} · n_emb={N_EMB}")
    print(BAR)

    if not args.no_source_check:
        section_source_check()
        print(BAR)

    rng = random.Random(args.seed)
    old_rate, new_rate = section_infeasible(args.draws, rng, OLD_SPACE, NEW_SPACE)
    print(BAR)
    section_cardinality(OLD_SPACE, NEW_SPACE)
    print(BAR)
    section_marginals(
        args.draws, random.Random(args.seed + 1), NEW_SPACE, "v5, antes de congelar"
    )
    print(BAR)
    bits_frozen = section_freeze(OLD_SPACE, NEW_SPACE)
    print(BAR)
    section_budget(NEW_SPACE)
    print(BAR)

    _say("=>", f"resumen: inviables {old_rate:.1%} -> {new_rate:.1%}")
    _say("=>", f"resumen: dimensionalidad efectiva congelada {bits_frozen:.1f} bits")
    if new_rate != 0.0:
        _say("[X]", "el muestreo v5 viola el gate estructural")
        return 1
    _say("[ok]", "simulacion completa")
    return 0


if __name__ == "__main__":
    sys.exit(main())
