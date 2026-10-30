#!/usr/bin/env python
"""Calibracion del gate de dominancia (select-v4) y auto-test de metricas.

Estudio autocontenido: genera particiones con estructura conocida, les aplica
ruido informativo (el que produce HDBSCAN: las clases densas pierden pocos
puntos, las pequenas pierden casi todos) y mide con que tasa cada forma de
gate clasifica la particion como degenerada / valida.

No importa `reii.main_workflow` (tarda >4 min por Stanza), no toca `data/`,
`output/` ni la BD, no usa red ni docker. Corre en pocos segundos.

Uso
---
    python scripts/playground/gate_calibration.py
    python scripts/playground/gate_calibration.py --reps 200
    python scripts/playground/gate_calibration.py --metrics-selftest

Salida: tablas ASCII con `->`, `[ok]`, `[!]`, `[X]`. Nada de box-drawing.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# ── Rejilla del estudio ─────────────────────────────────────────────────────
N_CORPUS = 528  # corpus real del proyecto
K_GRID = (3, 5, 8, 11, 14, 18, 20)
NU_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65)
K_BOUNDS = (3, 20)  # Config.method_selection_k_bounds

# Bandas declaradas de la VERDAD generativa (p1 = proporcion de la clase mayor
# en el corpus, ANTES del ruido). El hueco (0.65, 0.75) se reporta aparte y no
# se puntua: la simulacion no puede decidir donde esta el limite real, solo
# comprobar si la regla mantiene el veredicto al variar el ruido.
BAND_VALID = (0.30, 0.65)
BAND_DEGEN = (0.75, 0.92)
BAND_AMBIG = (0.65, 0.75)

TRUTH_VALID, TRUTH_DEGEN, TRUTH_AMBIG = 0, 1, 2

#: Nivel del gate. Punto medio del hueco declarado (0.65, 0.75). La
#: simulacion no identifica el valor: solo comprueba que la meseta admisible
#: (fv <= 1% y fd <= 0.5%) es T0 en [0.64, 0.70], y 0.70 es el extremo mas
#: permisivo de esa meseta. Ver GATE_NOISE.md, seccion 7.
T0 = 0.70

#: Forma recomendada. "corpus" -> c = a*(1-nu) <= T0 (dominancia sobre el
#: corpus, umbral constante). Ver GATE_NOISE.md para la derivacion.
RECOMMENDED = "corpus"


# ══════════════════════════════════════════════════════════════════════════
# 1. Generador
# ══════════════════════════════════════════════════════════════════════════
def _proportions(rng: np.random.Generator, k: int, family: int) -> np.ndarray:
    """Proporciones del corpus (suman 1) con la clase mayor controlada."""
    if family == TRUTH_VALID:
        for _ in range(500):
            p = rng.dirichlet(np.ones(k))
            if p.max() <= BAND_VALID[1] and p.min() >= 0.005:
                return np.sort(p)[::-1]
        p = rng.dirichlet(np.ones(k))
        return np.sort(p / p.sum())[::-1]
    if family == TRUTH_DEGEN:
        p1 = float(rng.uniform(*BAND_DEGEN))
    else:
        p1 = float(rng.uniform(*BAND_AMBIG))
    rest = rng.dirichlet(np.ones(k - 1)) * (1.0 - p1)
    return np.sort(np.concatenate([[p1], rest]))[::-1]


def dropout_rates(p: np.ndarray, nu: float) -> np.ndarray:
    """Probabilidad de ser ruido (-1) por clase, con nu global EXACTO.

    Modelo de umbral: cada UCE tiene un "score de densidad" p_clase * U con
    U ~ Uniforme(0,1); se descarta el nu% de scores mas bajos. Entonces
        q_j = min(1, t / p_j),
    con t resuelto para que sum_j p_j * q_j = nu. Las clases con p_j <= t
    desaparecen enteras: es el comportamiento de HDBSCAN con clases pequenas.
    """
    p = np.asarray(p, float)
    if nu <= 0.0:
        return np.zeros_like(p)
    ps = np.sort(p)[::-1]
    k = len(ps)
    tail = np.concatenate([np.cumsum(ps[::-1])[::-1], [0.0]])
    t = None
    for h in range(1, k + 1):
        cand = (nu - tail[h]) / h
        if cand <= 0.0:
            continue
        if cand <= ps[h - 1] and (h == k or cand >= ps[h]):
            t = cand
            break
    if t is None:
        t = nu / k
    return np.clip(t / p, 0.0, 1.0)


def _hill(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) < 2:
        return float(len(p))
    return float(np.exp(-(p * np.log(p)).sum()))


def simulate(rng: np.random.Generator, k: int, nu: float, family: int) -> dict:
    """Una particion cruda observada, con su verdad generativa."""
    p = _proportions(rng, k, family)
    sizes = rng.multinomial(N_CORPUS, p)
    q = dropout_rates(p, nu)
    assigned = rng.binomial(sizes, 1.0 - q)
    m = int(assigned.sum())
    if m <= 0:
        return {"ok": False}
    s1 = int(assigned.max())
    k_obs = int((assigned > 0).sum())
    prop = assigned[assigned > 0] / m
    return {
        "ok": True,
        "family": family,
        "k": k_obs,
        "k_gen": k,
        "nu": 1.0 - m / N_CORPUS,  # ruido REALIZADO (lo que ve la regla)
        "nu_nom": nu,  # ruido pedido (celda de la rejilla)
        "a": s1 / m,  # max_class_share (asignadas)
        "c": s1 / N_CORPUS,  # dominancia sobre el corpus
        "keff": _hill(prop),  # numero efectivo de clases (Hill)
        "p1": float(p.max()),
    }


def build_sample(seed: int, reps: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for family in (TRUTH_VALID, TRUTH_DEGEN, TRUTH_AMBIG):
        for k in K_GRID:
            for nu in NU_GRID:
                for _ in range(reps):
                    rec = simulate(rng, k, nu, family)
                    if rec["ok"]:
                        rows.append(rec)
    keys = ("family", "k", "k_gen", "nu", "nu_nom", "a", "c", "keff", "p1")
    return {key: np.array([r[key] for r in rows], float) for key in keys}


# ══════════════════════════════════════════════════════════════════════════
# 2. Formas del gate (True = descartar por degenerada)
# ══════════════════════════════════════════════════════════════════════════
def gate_actual(a, c, nu, keff, t0=T0):
    """Regla vigente: max_class_share > 0.90 (sobre asignadas)."""
    return a > 0.90


def gate_assigned(a, c, nu, keff, t0=T0):
    """Umbral constante sobre lo asignado: a > t0."""
    return a > t0


def gate_mixed(a, c, nu, keff, t0=T0):
    """Compromiso: a > t0 * (1-nu)^(-0.5)."""
    return a > t0 * np.power(np.maximum(1e-9, 1.0 - nu), -0.5)


def gate_corpus(a, c, nu, keff, t0=T0):
    """Umbral constante sobre el corpus: c = a*(1-nu) > t0."""
    return c > t0


def gate_tail(a, c, nu, keff, t0=T0):
    """Masa NO dominante del corpus por debajo de (1-t0)."""
    return (1.0 - nu - c) < (1.0 - t0)


def gate_keff(a, c, nu, keff, t0=T0):
    """Numero efectivo de clases (Hill) por debajo de 2."""
    return keff < 2.0


GATES = (
    ("actual  a>0.90", gate_actual),
    ("asignada a>0.70", gate_assigned),
    ("mixta   a>0.70/(1-nu)^0.5", gate_mixed),
    ("corpus  c>0.70", gate_corpus),
    ("tail    1-nu-c<0.30", gate_tail),
    ("keff    Hill<2", gate_keff),
)


# ══════════════════════════════════════════════════════════════════════════
# 3. Evaluacion
# ══════════════════════════════════════════════════════════════════════════
def _k_ok(sample: dict) -> np.ndarray:
    return (sample["k"] >= K_BOUNDS[0]) & (sample["k"] <= K_BOUNDS[1])


def rates(sample: dict, rejected: np.ndarray, only_k_ok: bool = False) -> dict:
    """Tasas de error contra la verdad generativa.

    ``only_k_ok=True`` aisla el gate de dominancia: solo cuenta particiones que
    llegan a el (K dentro de ``k_bounds``). Con ``False`` se reporta el
    comportamiento del pipeline completo, donde el gate de K tambien descarta.
    """
    fam = sample["family"]
    ok = _k_ok(sample) if only_k_ok else np.ones(len(fam), bool)
    rej = rejected | (~_k_ok(sample))  # K fuera de rango -> descartada
    valid = (fam == TRUTH_VALID) & ok
    degen = (fam == TRUTH_DEGEN) & ok
    fd = float(rej[valid].mean()) if valid.any() else float("nan")
    fv = float((~rej[degen]).mean()) if degen.any() else float("nan")
    return {
        "false_degen": fd,
        "false_valid": fv,
        "n_valid": int(valid.sum()),
        "n_degen": int(degen.sum()),
    }


def by_regime(sample, rejected, key="nu_nom", only_k_ok=False):
    out = {}
    for value in np.unique(sample[key]):
        mask = sample[key] == value
        sub = {k: v[mask] for k, v in sample.items()}
        out[float(value)] = rates(sub, rejected[mask], only_k_ok=only_k_ok)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 4. Presentacion
# ══════════════════════════════════════════════════════════════════════════
def _hdr(title: str) -> None:
    print("")
    print(title)
    print("-" * len(title))


def _mark(ok: bool) -> str:
    return "[ok]" if ok else "[X]"


def _cell(r: dict) -> str:
    fv, fd = r["false_valid"], r["false_degen"]
    if not np.isfinite(fv):
        return "-- /" + f"{fd:.2f}"
    if not np.isfinite(fd):
        return f"{fv:.2f}/-- "
    return f"{fv:.2f}/{fd:.2f}"


def main(reps: int) -> int:
    print("=" * 78)
    print("GATE CALIBRATION (select-v4) -- simulacion autocontenida")
    print(f"n_total={N_CORPUS}  reps={reps}  K={list(K_GRID)}")
    print(f"ruido nominal={list(NU_GRID)}")
    print(
        f"verdad: valida p1<={BAND_VALID[1]:.2f} | degenerada p1>={BAND_DEGEN[0]:.2f}"
        " | ambigua en medio (no se puntua)"
    )
    print(f"nivel del gate T0={T0:.2f} (punto medio del hueco declarado)")
    print("=" * 78)

    cal = build_sample(1000, reps)
    hold = build_sample(2000, reps)
    real_nu = {}
    for nu in NU_GRID:
        m = np.isclose(hold["nu_nom"], nu)
        real_nu[nu] = float(hold["nu"][m].mean()) if m.any() else float("nan")

    # ── [1] Comparacion de formas al mismo nivel ───────────────────────
    _hdr("[1] FORMAS DEL GATE al mismo nivel T0 (muestra de validacion)")
    print("fv = falso-valida (se acepta una degenerada)")
    print("fd = falso-degen (se descarta una valida)")
    print(
        f"{'forma':28}{'fv(pipeline)':>14}{'fd(pipeline)':>14}"
        f"{'fv(gate)':>11}{'fd(gate)':>10}"
    )
    for name, fn in GATES:
        rej = fn(hold["a"], hold["c"], hold["nu"], hold["keff"])
        r_all = rates(hold, rej, only_k_ok=False)
        r_gate = rates(hold, rej, only_k_ok=True)
        print(
            f"{name:28}{r_all['false_valid']:>14.3f}{r_all['false_degen']:>14.3f}"
            f"{r_gate['false_valid']:>11.3f}{r_gate['false_degen']:>10.3f}"
        )
    print("  (gate) aisla la dominancia: solo particiones con K en [3, 20]")

    # ── [2] Error por regimen de ruido, solo gate ──────────────────────
    _hdr("[2] ERROR POR RUIDO, solo gate de dominancia (fv/fd)")
    print("nu nominal ->" + "".join(f"{nu:>10.2f}" for nu in NU_GRID))
    print("nu real    ->" + "".join(f"{real_nu[nu]:>10.3f}" for nu in NU_GRID))
    for name, fn in GATES:
        rej = fn(hold["a"], hold["c"], hold["nu"], hold["keff"])
        reg = by_regime(hold, rej, "nu_nom", only_k_ok=True)
        cells = "".join(
            f"{_cell(reg[float(nu)]):>10}" if float(nu) in reg else f"{'--':>10}"
            for nu in NU_GRID
        )
        print(f"{name:28}{cells}")

    # ── [3] Cuantas particiones llegan al gate ─────────────────────────
    _hdr("[3] LLEGADA AL GATE: fraccion con K en [3, 20]")
    print("nu nominal ->" + "".join(f"{nu:>10.2f}" for nu in NU_GRID))
    for fam, label in ((TRUTH_VALID, "validas"), (TRUTH_DEGEN, "degeneradas")):
        cells = []
        for nu in NU_GRID:
            mask = (hold["nu_nom"] == nu) & (hold["family"] == fam)
            cells.append(
                f"{_k_ok(hold)[mask].mean():>10.2f}" if mask.any() else f"{'--':>10}"
            )
        print(f"{label:28}" + "".join(cells))
    print("  el resto lo descarta k_bounds antes de llegar a dominancia")

    # ── [4] Sensibilidad al nivel ──────────────────────────────────────
    _hdr("[4] SENSIBILIDAD AL NIVEL (forma recomendada, solo gate)")
    print(f"{'T0':>6}{'fv':>9}{'fd':>9}{'peor celda':>13}   veredicto")
    for cand in np.round(np.arange(0.58, 0.811, 0.02), 2):
        rej = gate_corpus(hold["a"], hold["c"], hold["nu"], hold["keff"], cand)
        r = rates(hold, rej, only_k_ok=True)
        reg = by_regime(hold, rej, "nu_nom", only_k_ok=True)
        worst = max(max(x["false_valid"], x["false_degen"]) for x in reg.values())
        flag = "  <- elegido" if math.isclose(cand, T0, abs_tol=1e-9) else ""
        print(
            f"{cand:>6.2f}{r['false_valid']:>9.3f}{r['false_degen']:>9.3f}"
            f"{worst:>13.3f}   {_mark(r['false_valid'] <= 0.01)}{flag}"
        )

    # ── [5] Casos medidos (probe v4) ───────────────────────────────────
    _hdr("[5] CASOS DEL PROBE v4 (misma estructura, distinto ruido)")
    cases = [
        ("denso sin ruido", [442, 45, 30, 11], 0),
        ("HDBSCAN 11.9% ruido", [442, 23], 63),
        ("HDBSCAN  4.9% ruido", [476, 26], 26),
    ]
    print(f"{'caso':22}{'a':>6}{'nu':>7}{'c':>6}   {'0.90':>8}{'corpus':>8}  veredicto")
    for label, sizes, n_noise in cases:
        n = sum(sizes) + n_noise
        m = sum(sizes)
        a = max(sizes) / m if m else float("nan")
        nu = n_noise / n
        c = max(sizes) / n
        one_a, one_c = np.array([a]), np.array([c])
        one_nu = np.array([nu])
        r_old = bool(gate_actual(one_a, one_c, one_nu, np.array([3.0]))[0])
        r_new = bool(gate_corpus(one_a, one_c, one_nu, np.array([3.0]))[0])
        verdict = (
            "ambas descartan"
            if r_old and r_new
            else "ambas aceptan"
            if (not r_old and not r_new)
            else "DISCREPAN"
        )
        print(
            f"{label:22}{a:>6.3f}{nu:>7.3f}{c:>6.3f}   "
            f"{'descarta' if r_old else 'acepta':>8} "
            f"{'descarta' if r_new else 'acepta':>8}  {verdict}"
        )

    # ── [6] Tabla de umbrales ──────────────────────────────────────────
    _hdr("[6] UMBRAL EQUIVALENTE SOBRE max_class_share: a <= T0/(1-nu)")
    print("Gate sobre el corpus (c <= 0.70). Leido sobre la variable que hoy")
    print("reporta el codigo, el umbral se relaja con el ruido: es la misma")
    print("regla, no una regla distinta por regimen.")
    print(f"{'nu':>6}{'nu real':>9}{'umbral c':>11}{'umbral a':>10}   efecto")
    for nu in NU_GRID:
        ta = T0 / (1.0 - nu)
        effect = "sin restriccion" if ta >= 1.0 else "activo"
        print(f"{nu:>6.2f}{real_nu[nu]:>9.3f}{T0:>11.2f}{ta:>10.3f}   {effect}")

    # ── [7] Error por K ────────────────────────────────────────────────
    _hdr("[7] ERROR POR K (forma recomendada, solo gate)")
    rej = gate_corpus(hold["a"], hold["c"], hold["nu"], hold["keff"])
    print(f"{'K':>4}{'fv':>9}{'fd':>9}{'n(gate)':>10}")
    for k in K_GRID:
        mask = np.isclose(hold["k_gen"], k)
        sub = {kk: v[mask] for kk, v in hold.items()}
        if not len(sub["a"]):
            continue
        r = rates(sub, rej[mask], only_k_ok=True)
        print(
            f"{k:>4}{r['false_valid']:>9.3f}{r['false_degen']:>9.3f}"
            f"{r['n_valid'] + r['n_degen']:>10}"
        )

    # ── [8] Piso de ARI de una particion dominante ────────────────────
    _hdr("[8] PISO DE ARI: predictor trivial que solo acierta la clase mayor")
    from sklearn.metrics import adjusted_rand_score as _ari

    print("obs = copia exacta de la clase mayor + resto al azar (K=3 y K=8)")
    print(f"{'c':>6}{'ARI_azar':>10}{'ARI_K3':>9}{'ARI_K8':>9}")
    _rng = np.random.default_rng(0)
    for c in (0.50, 0.65, 0.70, 0.75, 0.80, 0.837, 0.90, 0.95):
        n1 = int(round(c * 528))
        values = []
        for s in (2, 7):
            sat = [(528 - n1) // s] * s
            sat[0] += (528 - n1) - sum(sat)
            ref = np.concatenate([np.full(x, i) for i, x in enumerate([n1] + sat)])
            fake, rand = [], []
            for _ in range(100):
                rand.append(_ari(ref, _rng.integers(0, s + 1, size=528)))
                obs = _rng.integers(0, s + 1, size=528)
                obs[:n1] = 0
                fake.append(_ari(ref, obs))
            values.append((float(np.mean(rand)), float(np.mean(fake))))
        print(f"{c:>6.3f}{values[0][0]:>10.3f}{values[0][1]:>9.3f}{values[1][1]:>9.3f}")
    print("  azar puro = 0 por construccion del ARI; el piso viene de la")
    print("  clase mayor: a c=0.837 un predictor trivial ya cobra ARI~0.72 (K=3)")

    # ── [9] Ruta del nulo de composicion ──────────────────────────────
    _hdr("[9] RUTA 2: umbral = cuantil del nulo de max_class_share")
    print("nulo Dirichlet(1) con el K observado; n=528")
    print(f"{'K':>3}{'q50':>8}{'q95':>8}{'q99':>8}   veredicto a c=0.837")
    for k in K_GRID:
        comp = np.random.default_rng(0).dirichlet(np.ones(k), size=20000)
        share = comp.max(axis=1)
        q95 = float(np.quantile(share, 0.95))
        q99 = float(np.quantile(share, 0.99))
        verdict = "acepta (no detecta el bug)" if q95 > 0.837 else "descarta"
        print(
            f"{k:>3}{np.quantile(share, 0.50):>8.3f}{q95:>8.3f}{q99:>8.3f}"
            f"   q95 -> {verdict}"
        )
    print("  el umbral nulo varia 0.87 -> 0.27 segun K: es un segundo gate de K")

    print("")
    print("=" * 78)
    print("LECTURA")
    print("1. El 0.90 constante sobre lo asignado tiene fv alto SIN ruido: con")
    print("   denominador completo la clase gigante se diluye y pasa el gate.")
    print("2. Ese mismo 0.90 equivale, sobre el corpus, a exigir c <= 0.90(1-nu),")
    print("   es decir un umbral que YA depende del ruido, sin declararlo.")
    print("3. Bajar el umbral con el ruido (a > T0*(1-nu)) anade una segunda")
    print("   dosis de la misma correccion y produce fd masivo: es dominado.")
    print("4. La forma recomendada fija el nivel sobre el CORPUS (c <= 0.70),")
    print("   que es la cantidad que infla el ARI, y deja que el umbral sobre")
    print("   lo asignado se relaje como 0.70/(1-nu).")
    print("5. El piso de ARI de [8] muestra POR QUE el gate no puede ser un")
    print("   umbral de ARI: con c alto el ARI sube sin informacion nueva.")
    print("6. La ruta del nulo de composicion [9] da un umbral que varia con K")
    print("   (0.87 a K=3, 0.27 a K=20): no detecta el caso 0.837 y es un")
    print("   segundo gate de K disfrazado.")
    print("=" * 78)
    return 0


# ══════════════════════════════════════════════════════════════════════════
# 5. Auto-test de las metricas prometidas (C_in, d, RMSD)
# ══════════════════════════════════════════════════════════════════════════
# PROTOTIPO: copia literal de las funciones que se proponen para
# `src/reii/quality/metrics.py`. Se usan aqui para poder correr el estudio
# antes de aplicar el diff. Cuando el diff este aplicado, el script prefiere
# las funciones del modulo y estas quedan sin uso.
def _row_normalize(X):
    from scipy import sparse

    if sparse.issparse(X):
        X = X.tocsr().astype(float)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        return sparse.diags(1.0 / np.where(norms > 0, norms, 1.0)) @ X
    X = np.asarray(X, float)
    norms = np.linalg.norm(X, axis=1)
    return X / np.where(norms > 0, norms, 1.0)[:, None]


def _cos_to_centroid(Xn, y, classes):
    """(n_c, coseno medio al centroide) por clase, cosenos truncados, n negativos."""
    per_class = []
    per_item = []
    n_neg = 0
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        if len(idx) == 0:
            continue
        mu = np.asarray(Xn[idx].mean(axis=0)).ravel()
        norm = float(np.linalg.norm(mu))
        cos = (
            np.zeros(len(idx))
            if norm <= 0
            else np.asarray(Xn[idx] @ (mu / norm)).ravel()
        )
        n_neg += int((cos < 0.0).sum())
        cos = np.clip(cos, 0.0, 1.0)
        per_class.append((len(idx), float(cos.mean())))
        per_item.extend(cos.tolist())
    return per_class, np.asarray(per_item, float), n_neg


def _bootstrap_ci(values, stat=np.mean, n_boot=500, alpha=0.05, seed=42):
    v = np.asarray(values, float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array(
        [stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    )
    return (
        float(stat(v)),
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def internal_coherence(
    matrix,
    labels,
    space_id,
    metric="cosine",
    max_pairwise=1500,
    n_boot=500,
    n_perm=500,
    seed=42,
):
    """C_in: homogeneidad intra-clase en un espacio declarado."""
    from scipy import sparse

    y = np.asarray(list(labels)).astype(int)
    X_full = matrix if hasattr(matrix, "shape") else np.asarray(matrix, float)
    if X_full.shape[0] != len(y):
        raise ValueError("matrix y labels deben tener el mismo numero de filas")
    keep = y >= 0
    X = X_full[keep] if sparse.issparse(X_full) else np.asarray(X_full, float)[keep]
    y = y[keep]
    n = int(len(y))
    classes = np.unique(y)
    k = int(len(classes))
    out = {
        "cin": float("nan"),
        "cin_weighted": float("nan"),
        "cin_pairwise": float("nan"),
        "space_id": space_id,
        "metric": metric,
        "n": n,
        "k": k,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "ci_low_weighted": float("nan"),
        "ci_high_weighted": float("nan"),
        "frac_negative": float("nan"),
        "null_mean": float("nan"),
        "null_std": float("nan"),
        "excess": float("nan"),
        "z_score": float("nan"),
        "p_value": float("nan"),
    }
    if n < 3 or k < 2:
        return out
    Xn = _row_normalize(X)
    per_class, per_item, n_neg = _cos_to_centroid(Xn, y, classes)
    if not per_class:
        return out
    coin = np.array([m for _, m in per_class])
    counts = np.array([c for c, _ in per_class], float)
    cin = float(coin.mean())
    out["cin"] = cin
    out["cin_weighted"] = float((counts * coin).sum() / counts.sum())
    out["frac_negative"] = n_neg / float(n)
    out["ci_low"], out["ci_high"] = _bootstrap_ci(
        coin, np.mean, n_boot=n_boot, seed=seed
    )[1:]
    out["ci_low_weighted"], out["ci_high_weighted"] = _bootstrap_ci(
        per_item, np.mean, n_boot=n_boot, seed=seed
    )[1:]
    # pairwise (opcional, O(n_c^2) por clase)
    pair = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        if len(idx) < 2:
            continue
        if len(idx) > max_pairwise:
            idx = np.random.default_rng(seed).choice(idx, max_pairwise, replace=False)
        sim = (
            np.asarray((Xn[idx] @ Xn[idx].T).todense())
            if hasattr(Xn[idx] @ Xn[idx].T, "todense")
            else np.asarray(Xn[idx] @ Xn[idx].T)
        )
        iu = np.triu_indices(len(idx), 1)
        pair.append(float(np.clip(sim[iu], 0.0, 1.0).mean()))
    if pair:
        out["cin_pairwise"] = float(np.mean(pair))
    # nulo: permutacion de etiquetas (rompe la asociacion UCE-clase)
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        pc, _, _ = _cos_to_centroid(Xn, rng.permutation(y), classes)
        nulls[i] = float(np.mean([m for _, m in pc]))
    std = float(nulls.std(ddof=1))
    out["null_mean"] = float(nulls.mean())
    out["null_std"] = std
    out["excess"] = cin - out["null_mean"]
    out["z_score"] = (cin - out["null_mean"]) / std if std > 0 else float("nan")
    out["p_value"] = float((nulls >= cin).mean())
    return out


def _entropy_stats(y, k_ref=None):
    counts = np.bincount(y)
    counts = counts[counts > 0]
    n = int(counts.sum())
    p = counts / n
    h = float(-(p * np.log(p)).sum())
    k = int(len(p))
    k_norm = int(k_ref) if k_ref else k
    return {
        "n": n,
        "k": k,
        "entropy": h,
        "entropy_norm": h / np.log(k_norm) if k_norm > 1 else float("nan"),
        "simpson": float(1.0 - (p * p).sum()),
        "effective_k": float(np.exp(h)),
        "max_share": float(p.max()),
    }


def partition_diversity(labels, n_boot=500, n_null=500, seed=42):
    """d: diversidad de la particion (entropia normalizada, Simpson, Hill)."""
    y = np.asarray(list(labels)).astype(int)
    y = y[y >= 0]
    base = _entropy_stats(y) if len(y) else None
    out = (
        dict(base)
        if base
        else {
            "n": 0,
            "k": 0,
            "entropy": float("nan"),
            "entropy_norm": float("nan"),
            "simpson": float("nan"),
            "effective_k": float("nan"),
            "max_share": float("nan"),
        }
    )
    out.update(
        {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "null_mean": float("nan"),
            "null_q05": float("nan"),
            "null_q95": float("nan"),
            "p_value": float("nan"),
            "label_permutation_invariant": True,
            "space_id": "cluster_id",
        }
    )
    if base is None or base["k"] < 2:
        return out
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        # K fijo en el denominador: el bootstrap remuestrea UCEs, no clases
        boots[i] = _entropy_stats(
            rng.choice(y, size=len(y), replace=True), k_ref=base["k"]
        )["entropy_norm"]
    out["ci_low"] = float(np.quantile(boots, 0.025))
    out["ci_high"] = float(np.quantile(boots, 0.975))
    # nulo de COMPOSICION: Dirichlet(1) con el mismo n y K. La permutacion de
    # etiquetas no sirve aqui: d depende solo de las frecuencias, no de quien
    # esta en cada clase, y es invariante a permutar etiquetas.
    comp = rng.dirichlet(np.ones(base["k"]), size=n_null)
    h = -(comp * np.log(comp)).sum(axis=1) / np.log(base["k"])
    out["null_mean"] = float(h.mean())
    out["null_q05"] = float(np.quantile(h, 0.05))
    out["null_q95"] = float(np.quantile(h, 0.95))
    p_lo = float((h <= base["entropy_norm"]).mean())
    p_hi = float((h >= base["entropy_norm"]).mean())
    out["p_value"] = float(min(1.0, 2.0 * min(p_lo, p_hi)))
    return out


def _rmsd_core(ra, ob):
    from scipy.optimize import linear_sum_assignment

    ru, ri = np.unique(ra, return_inverse=True)
    ou, oi = np.unique(ob, return_inverse=True)
    n = len(ra)
    ov = np.zeros((len(ru), len(ou)))
    np.add.at(ov, (ri, oi), 1.0)
    row, col = ov.sum(axis=1), ov.sum(axis=0)
    r, c = linear_sum_assignment(-ov)
    pairs = [(row[i] / n, col[j] / n) for i, j in zip(r, c)]
    used_r, used_c = set(r.tolist()), set(c.tolist())
    pairs += [(row[i] / n, 0.0) for i in range(len(ru)) if i not in used_r]
    pairs += [(0.0, col[j] / n) for j in range(len(ou)) if j not in used_c]
    pr = np.array([p for p, _ in pairs])
    po = np.array([q for _, q in pairs])
    return {
        "rmsd": float(np.sqrt(np.mean((pr - po) ** 2))) if len(pr) else float("nan"),
        "k_ref": int(len(ru)),
        "k_obs": int(len(ou)),
        "k_union": int(len(pr)),
        "n_unmatched": int(len(pr) - len(r)),
    }


def rmsd_proportions(
    labels_ref, labels_obs, reference_id, n_boot=300, n_null=300, seed=42
):
    """RMSD entre proporciones de clase, con emparejamiento Hungarian."""
    common = [
        u
        for u in sorted(set(labels_ref) & set(labels_obs))
        if labels_ref[u] is not None
        and labels_obs[u] is not None
        and int(labels_ref[u]) >= 0
        and int(labels_obs[u]) >= 0
    ]
    out = {
        "rmsd": float("nan"),
        "n": len(common),
        "k_ref": 0,
        "k_obs": 0,
        "k_union": 0,
        "n_unmatched": 0,
        "reference_id": reference_id,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "null_mean": float("nan"),
        "null_std": float("nan"),
        "null_q05": float("nan"),
        "null_q95": float("nan"),
        "p_value": float("nan"),
    }
    if len(common) < 2:
        return out
    ra = np.array([int(labels_ref[u]) for u in common])
    ob = np.array([int(labels_obs[u]) for u in common])
    core = _rmsd_core(ra, ob)
    out.update(core)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(common), len(common))
        boots[i] = _rmsd_core(ra[idx], ob[idx])["rmsd"]
    out["ci_low"] = float(np.quantile(boots, 0.025))
    out["ci_high"] = float(np.quantile(boots, 0.975))
    # nulo: permutar las etiquetas observadas entre las UCEs comunes. Mantiene
    # los margenes y rompe la asociacion; el emparejamiento Hungarian cambia y
    # el RMSD empeora si el acuerdo observado era real.
    nulls = np.empty(n_null)
    for i in range(n_null):
        nulls[i] = _rmsd_core(ra, rng.permutation(ob))["rmsd"]
    out["null_mean"] = float(nulls.mean())
    out["null_std"] = float(nulls.std(ddof=1))
    out["null_q05"] = float(np.quantile(nulls, 0.05))
    out["null_q95"] = float(np.quantile(nulls, 0.95))
    out["p_value"] = float((nulls >= out["rmsd"]).mean())
    return out


def _load_workflow_data():
    import json

    path = ROOT / "data" / "workflow_data.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_metrics_selftest() -> int:
    data = _load_workflow_data()
    if data is None:
        print("[!] data/workflow_data.json no encontrado: auto-test omitido")
        return 1
    uces = data.get("uces") or []
    keep = [u for u in uces if u.get("embedding") is not None]
    n_skip = len(uces) - len(keep)
    if len(keep) < 10:
        print("[!] menos de 10 UCEs con embedding: auto-test omitido")
        return 1
    X = np.asarray([u["embedding"] for u in keep], float)
    ids = [str(u.get("id")) for u in keep]
    cid = np.array([int(u.get("cluster_id", -1)) for u in keep], int)
    obs = {i: int(v) for i, v in zip(ids, cid)}

    # Este bench no trae `projected_cluster_id` (todo None), asi que la
    # referencia del RMSD se construye aqui: una particion ALTERNATIVA del
    # mismo espacio (k-means con el mismo K). Sirve para ilustrar el calculo,
    # NO como validez externa (no hay particion humana/oraculo).
    from sklearn.cluster import KMeans

    k_obs = int(len(np.unique(cid[cid >= 0])))
    km = KMeans(n_clusters=max(2, k_obs), n_init=10, random_state=42).fit_predict(X)
    ref = {i: int(v) for i, v in zip(ids, km)}
    ref_id = f"kmeans_k{k_obs}_sobre_uce_embedding_384d"

    print("")
    print("=" * 78)
    print("AUTO-TEST DE METRICAS sobre data/workflow_data.json")
    print(f"(bench uce=26: {len(uces)} UCEs, NO es la corrida en curso)")
    print(f"(con embedding: {len(keep)}; sin embedding (None): {n_skip})")
    print("=" * 78)

    cin = internal_coherence(X, cid, "uce_embedding_384d", n_boot=300, n_perm=300)
    print("")
    print("C_in  (coseno al centroide de clase, macro, clip a [0,1])")
    print(f"  espacio={cin['space_id']}  n={cin['n']}  k={cin['k']}")
    print(f"  C_in={cin['cin']:.4f}   IC95=[{cin['ci_low']:.4f}, {cin['ci_high']:.4f}]")
    print(
        f"  ponderado={cin['cin_weighted']:.4f}"
        f"  IC95=[{cin['ci_low_weighted']:.4f}, {cin['ci_high_weighted']:.4f}]"
    )
    print(
        f"  pairwise={cin['cin_pairwise']:.4f}"
        f"  cosenos negativos truncados={cin['frac_negative']:.4f}"
    )
    print(
        f"  nulo(permutacion de etiquetas)={cin['null_mean']:.4f}"
        f"  exceso={cin['excess']:+.4f}  z={cin['z_score']:.2f}"
        f"  p={cin['p_value']:.3f}"
    )
    if cin["p_value"] >= 0.05:
        print("  [!] C_in NO se distingue de un corpus homogeneo con estas clases")
    else:
        print("  [ok] C_in supera el nulo de etiquetas permutadas")

    div = partition_diversity(cid, n_boot=300, n_null=300)
    print("")
    print("d   (diversidad de la particion; NO es diversidad semantica por UCE)")
    print(
        f"  n={div['n']}  k={div['k']}  H/lnK={div['entropy_norm']:.4f}"
        f"  IC95=[{div['ci_low']:.4f}, {div['ci_high']:.4f}]"
    )
    print(
        f"  Simpson={div['simpson']:.4f}  K_eff(Hill)={div['effective_k']:.2f}"
        f"  max_share={div['max_share']:.4f}"
    )
    print(
        f"  nulo(composicion Dirichlet(1))={div['null_mean']:.4f}"
        f"  q05={div['null_q05']:.4f}  q95={div['null_q95']:.4f}"
        f"  p={div['p_value']:.3f}"
    )

    rmsd = rmsd_proportions(ref, obs, ref_id, n_boot=200, n_null=200)
    print("")
    print("RMSD (proporciones; referencia declarada = particion k-means)")
    print(f"  referencia={rmsd['reference_id']}")
    print(
        f"  n={rmsd['n']}  k_ref={rmsd['k_ref']}  k_obs={rmsd['k_obs']}"
        f"  k_union={rmsd['k_union']}"
    )
    print(
        f"  RMSD={rmsd['rmsd']:.4f}  IC95=[{rmsd['ci_low']:.4f}, {rmsd['ci_high']:.4f}]"
    )
    print(
        f"  nulo(permutacion de etiquetas)={rmsd['null_mean']:.4f}"
        f"  sd={rmsd['null_std']:.4f}  q95={rmsd['null_q95']:.4f}"
        f"  p={rmsd['p_value']:.3f}"
    )
    if rmsd["p_value"] >= 0.05:
        print(
            "  [!] RMSD indistinguible del nulo: proporciones parecidas NO"
            " son evidencia de acuerdo entre particiones"
        )
    else:
        print("  [ok] RMSD por debajo del nulo de etiquetas permutadas")
    # Diagnostico de invariancia: misma particion pero SIN asociacion real
    # (se conserva el vector de tamanos). Si el RMSD no baja con el acuerdo,
    # lo que mide es el desajuste de perfiles.
    _sizes = sorted(Counter(obs.values()).values(), reverse=True)
    _lab = np.concatenate([np.full(s, i) for i, s in enumerate(_sizes)])
    np.random.default_rng(7).shuffle(_lab)
    _rand = {u: int(v) for u, v in zip(ids, _lab)}
    _r = rmsd_proportions(ref, _rand, "aleatoria_mismo_perfil", n_boot=50, n_null=50)
    print(
        f"  control: RMSD contra particion ALEATORIA con el mismo perfil"
        f" = {_r['rmsd']:.4f}  (observado {rmsd['rmsd']:.4f})"
    )
    if _r["rmsd"] <= rmsd["rmsd"]:
        print(
            "  [!] el azar con el mismo perfil da RMSD igual o menor:"
            " el estadistico no distingue acuerdo de perfil"
        )
    else:
        print(
            "  [i] el acuerdo real mejora el perfil, pero dentro del rango"
            " del nulo (ver sd)"
        )
    print("  [i] sin referencia EXTERNA esto es concordancia de proporciones")
    print("      entre dos salidas del pipeline, no validez externa")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reps", type=int, default=120, help="replicas por celda (K x ruido x familia)"
    )
    ap.add_argument(
        "--metrics-selftest",
        action="store_true",
        help="ademas, corre C_in / d / RMSD sobre data/workflow_data.json",
    )
    args = ap.parse_args()
    rc = main(args.reps)
    if args.metrics_selftest:
        rc |= run_metrics_selftest()
    sys.exit(rc)
