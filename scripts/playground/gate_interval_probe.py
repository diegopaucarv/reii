#!/usr/bin/env python
"""Probe de identificabilidad del gate de dominancia (select-v4) y del RMSD.

Responde tres preguntas con numeros, no con opinion:

  [1] IDENTIFICABILIDAD.  Con ruido informativo solo se observa
      ``c = p1 * (1 - q1)``.  El conjunto identificado de ``p1`` (clase mayor
      del corpus ANTES del ruido) es ``[c, min(1, c + nu)]`` y su ancho es
      ``min(nu, 1 - c)``: la resolucion del gate es el presupuesto de ruido.
      Se cuantifica el ancho por regimen de ``nu`` y ``p1`` y se cuenta en
      cuantas celdas el gate acepta mientras el intervalo aun admite un
      corpus degenerado (``p1 >= 0.75``).

  [2] VACUIDAD.  Con ``nu >= 1 - C0 = 0.30`` se cumple ``c <= 1 - nu <= C0``
      para TODA particion, de modo que el gate no puede rechazar nada.  Se
      mide ademas su alcance real: solo puede disparar si
      ``a > C0 / (1 - nu)`` given ``a = c / (1 - nu)``.

  [3] RMSD.  El RMSD de proporciones (emparejamiento Hungarian) no mide
      acuerdo: si las dos particiones tienen el mismo multiconjunto de
      tamanos vale 0 exactamente, con independencia de la asociacion.  Se
      contrasta con ARI/NMI, que si separan.

No importa ``reii.main_workflow`` (Stanza, > 4 min).  No usa red, docker,
``data/`` ni la BD.  Salida ASCII puro (consola cp1252): ``->``, ``[ok]``,
``[!]``, ``[X]``.

Uso
---
    python scripts/playground/gate_interval_probe.py
    python scripts/playground/gate_interval_probe.py --reps 800
    python scripts/playground/gate_interval_probe.py --check-module

``--check-module`` compara el gate del probe con ``reii.method_selection``
importado del repo, si el diff de ``GATE_INTERVALS.md`` ya esta aplicado.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys

import numpy as np

# ── Parametros del estudio ─────────────────────────────────────────────────
N_CORPUS = 528  # corpus real del proyecto
C0 = 0.70  # nivel del gate (GATE_NOISE.md, seccion 6)
K_CLASSES = 5  # 1 clase mayor + 4 satelites de igual masa
P1_GRID = (0.35, 0.50, 0.60, 0.70, 0.72, 0.80, 0.90)
NU_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
REPS = 400

#: Verdad declarada en GATE_NOISE.md 6.1.  El hueco (0.65, 0.75) NO se puntua.
P1_VALID = 0.65
P1_DEGEN = 0.75

#: Tolerancia para comparar flotantes en las fronteras (nu = 0.30 exacto).
EPS = 1e-9


# ══════════════════════════════════════════════════════════════════════════
# 0. Regla propuesta (especificacion normativa del diff)
# ══════════════════════════════════════════════════════════════════════════
def profile(p1: float, k: int = K_CLASSES) -> np.ndarray:
    """Vector de proporciones con una clase mayor ``p1`` y satelites iguales."""
    rest = (1.0 - p1) / (k - 1)
    return np.array([p1] + [rest] * (k - 1), dtype=float)


def gate_dominance(metric) -> float:
    """``c = max_class_share * (1 - noise_frac)`` (fraccion del corpus)."""
    share = float(getattr(metric, "max_class_share", 0.0) or 0.0)
    noise = min(1.0, max(0.0, float(getattr(metric, "noise_frac", 0.0) or 0.0)))
    return share * (1.0 - noise)


def identified_interval(c: float, nu: float):
    """Conjunto identificado de ``p1``: ``[c, min(1, c + nu)]``.

    Cota inferior: toda clase pierde al menos 0 masa -> ``p1 >= c``.
    Cota superior: la clase mayor puede perder como maximo TODO el ruido
    -> ``p1 <= c + nu``.  No supone nada sobre el mecanismo de ruido.
    """
    lo = float(c)
    hi = min(1.0, float(c) + float(nu))
    return lo, hi


def interval_width(c: float, nu: float) -> float:
    """``min(nu, 1 - c)``: la resolucion del gate es el presupuesto de ruido."""
    return min(float(nu), 1.0 - float(c))


def gate_status(c: float, nu: float, c0: float = C0) -> str:
    """Estado evidencial del gate para UNA particion.

    Precedencia: ``reject`` > ``vacuous`` > ``certified`` > ``indeterminate``.

    ``reject``        c > C0           -> certifica p1 > C0 (solido)
    ``vacuous``       nu >= 1 - C0     -> el gate no puede rechazar nada
    ``certified``     c + nu <= C0     -> certifica p1 <= C0
    ``indeterminate`` resto            -> acepta pero no certifica
    """
    c = float(c)
    nu = float(nu)
    if c > float(c0) + EPS:
        return "reject"
    if nu >= 1.0 - float(c0) - EPS:
        return "vacuous"
    if c + nu <= float(c0) + EPS:
        return "certified"
    return "indeterminate"


# ══════════════════════════════════════════════════════════════════════════
# 1. Modelo de ruido (mismo que gate_calibration.py, seccion 1)
# ══════════════════════════════════════════════════════════════════════════
def dropout_rates(p: np.ndarray, nu: float) -> np.ndarray:
    """Tasa de ruido por clase con ``nu`` global EXACTO (dropout por umbral).

    Cada UCE tiene un score ``p_clase * U``, ``U ~ Uniforme(0,1)``; se
    descarta el ``nu`` % de scores mas bajos.  Resultado: ``q_j = min(1, t/p_j)``
    con ``t`` resuelto para que ``sum_j p_j q_j = nu``.  Es el ruido
    INFORMATIVO: las clases densas pierden poco, las pequenas desaparecen.
    Con ``q_1 ~ 0`` se cumple ``c ~ p1``.
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


def mc_cell(rng, p1: float, nu: float, model: str, reps: int = REPS):
    """Reparte el corpus y devuelve ``(c, nu_real, p1_real)`` por replica.

    ``nu_real = 1 - m/n`` es el ruido REALIZADO, que es el que ve la regla
    (``noise_frac`` se mide, no se supone).  ``p1_real = max(sizes)/n`` es la
    clase mayor realizada del corpus: para un corpus fijo el estimando no
    tiene error de muestreo, solo reetiquetado por ruido.
    """
    p = profile(p1)
    sizes = rng.multinomial(N_CORPUS, p, size=reps)
    if model == "informative":
        keep = 1.0 - dropout_rates(p, nu)
    else:  # uniforme: cada clase pierde la misma fraccion nu
        keep = np.full(len(p), 1.0 - nu)
    assigned = rng.binomial(sizes, keep)
    c = assigned.max(axis=1) / float(N_CORPUS)
    nu_real = 1.0 - assigned.sum(axis=1) / float(N_CORPUS)
    p1_real = sizes.max(axis=1) / float(N_CORPUS)
    k_obs = (assigned > 0).sum(axis=1)
    return c, nu_real, p1_real, k_obs


# ══════════════════════════════════════════════════════════════════════════
# 2. RMSD de proporciones (copia de la implementacion propuesta)
# ══════════════════════════════════════════════════════════════════════════
def _rmsd_core(ra: np.ndarray, ob: np.ndarray) -> dict:
    """RMSD con emparejamiento Hungarian que maximiza el solape."""
    ru, ri = np.unique(ra, return_inverse=True)
    ou, oi = np.unique(ob, return_inverse=True)
    n = len(ra)
    ov = np.zeros((len(ru), len(ou)))
    np.add.at(ov, (ri, oi), 1.0)
    row, col = ov.sum(axis=1), ov.sum(axis=0)
    if len(ru) * len(ou) <= 362880:  # hasta 9! -> fuerza bruta si no hay scipy
        try:
            from scipy.optimize import linear_sum_assignment

            r, c = linear_sum_assignment(-ov)
        except ImportError:  # pragma: no cover
            best, r, c = -1.0, np.arange(len(ru)), np.arange(len(ou))
            for perm in itertools.permutations(range(len(ou)), len(ru)):
                s = sum(ov[i, j] for i, j in zip(range(len(ru)), perm))
                if s > best:
                    best, r, c = s, np.arange(len(ru)), np.array(perm)
    else:  # pragma: no cover
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-ov)
    pairs = [(row[i] / n, col[j] / n) for i, j in zip(r, c)]
    used_r, used_c = (
        set(np.asarray(r).ravel().tolist()),
        set(np.asarray(c).ravel().tolist()),
    )
    pairs += [(row[i] / n, 0.0) for i in range(len(ru)) if i not in used_r]
    pairs += [(0.0, col[j] / n) for j in range(len(ou)) if j not in used_c]
    pr = np.array([a for a, _ in pairs], float)
    po = np.array([b for _, b in pairs], float)
    return {
        "rmsd": float(np.sqrt(np.mean((pr - po) ** 2))) if len(pr) else float("nan"),
        "k_ref": int(len(ru)),
        "k_obs": int(len(ou)),
    }


def _labels_from_sizes(sizes, rng=None):
    lab = np.concatenate([np.full(int(s), i) for i, s in enumerate(sizes)])
    if rng is not None:
        lab = lab[rng.permutation(len(lab))]
    return lab.astype(int)


def _perturb(lab: np.ndarray, frac: float, k: int, rng) -> np.ndarray:
    out = lab.copy()
    idx = rng.choice(len(out), size=max(1, int(round(frac * len(out)))), replace=False)
    out[idx] = rng.integers(0, k, size=len(idx))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 3. Impresion
# ══════════════════════════════════════════════════════════════════════════
def hr(char: str = "=", n: int = 78) -> None:
    print(char * n)


def _mark(p1: float, c: float, nu: float) -> str:
    """Marca de veredicto contra la verdad declarada (0.65 / 0.75)."""
    reject = c > C0 + EPS
    if p1 >= P1_DEGEN:
        return "[ok]" if reject else "[!]"
    if p1 <= P1_VALID:
        return "[X]" if reject else "[ok]"
    return "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=REPS, help="replicas por celda")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument(
        "--check-module",
        action="store_true",
        help="contrasta el gate del probe con reii.method_selection si el diff esta aplicado",
    )
    a = ap.parse_args(argv)
    reps = int(a.reps)
    rng = np.random.default_rng(a.seed)

    hr()
    print("PROBE DE INTERVALOS DEL GATE (select-v4) y del RMSD")
    print(
        "  n_corpus=%d  C0=%.2f  k=%d (1 mayor + %d satelites)  reps=%d"
        % (N_CORPUS, C0, K_CLASSES, K_CLASSES - 1, reps)
    )
    print("  sin docker, sin red, sin data/, sin reii.main_workflow")
    hr()

    # ── [1] Identificabilidad: tabla analitica de anchos ───────────────────
    print("")
    print("[1] INTERVALO IDENTIFICADO DE p1  y  SU ANCHO")
    print("    p1 in [c, min(1, c+nu)]   ->   ancho = min(nu, 1-c)")
    print("    tabla con c = p1 (q1=0, regimen de ruido informativo)")
    print("")
    print("    nu\\p1   " + "".join(f"{p:>7.2f}" for p in P1_GRID))
    print("    " + "-" * 64)
    for nu in NU_GRID:
        cells = "".join(f"{min(nu, 1.0 - p):>7.2f}" for p in P1_GRID)
        print(f"    {nu:>5.2f}   {cells}")
    print("")
    print("    lectura: a nu=0.20 el gate no distingue c=0.50 de p1=0.70;")
    print("    ni c=0.70 de p1=0.90.  El ancho es el presupuesto de ruido.")

    # ── [2] Alcance duro del gate ──────────────────────────────────────────
    print("")
    print("[2] ALCANCE DEL GATE: solo puede disparar si a > C0/(1-nu)")
    print("")
    print("    nu     c_max=1-nu   a_min=umbral   puede rechazar?")
    for nu in NU_GRID:
        c_max = 1.0 - nu
        a_min = C0 / c_max if c_max > 0 else float("inf")
        verdict = "[ok] si" if a_min < 1.0 - EPS else "[!] NO (vacuamente cierto)"
        print(f"    {nu:>4.2f}      {c_max:>8.2f}      {a_min:>8.3f}     {verdict}")
    print("")
    print(
        "    c <= m/n = 1-nu para toda particion, luego con nu >= 1-C0 = %.2f"
        % (1.0 - C0)
    )
    print("    el gate NO puede rechazar nada: vacuidad exacta, no estadistica.")

    # ── [3] Monte Carlo por celda ──────────────────────────────────────────
    rows = []
    for model in ("informative", "uniform"):
        print("")
        print(f"[3] MONTE CARLO por celda  (modelo: {model})")
        print("    celda = c medio; marca contra la verdad declarada")
        print("      [ok] acuerdo   [!] falso-valida  [X] falso-degenerada  - ambiguo")
        print("")
        print("    p1\\nu " + "".join(f"{nu:>9.2f}" for nu in NU_GRID))
        print("    " + "-" * 70)
        for p1 in P1_GRID:
            cells = []
            for nu in NU_GRID:
                c_real, nu_real, p1_real, _k = mc_cell(rng, p1, nu, model, reps)
                c = float(c_real.mean())
                nrm = float(nu_real.mean())
                rows.append(
                    {
                        "model": model,
                        "p1": p1,
                        "nu": nu,
                        "nu_real": nrm,
                        "c": c,
                        "sd": float(c_real.std(ddof=1)),
                        "width": float(
                            np.mean(
                                [interval_width(x, y) for x, y in zip(c_real, nu_real)]
                            )
                        ),
                        "reject": float((c_real > C0 + EPS).mean()),
                        "accept_degen": float(
                            np.mean(
                                (c_real <= C0 + EPS) & (c_real + nu_real >= P1_DEGEN)
                            )
                        ),
                        "covers": float(
                            np.mean(
                                (p1_real >= c_real - EPS)
                                & (p1_real <= np.minimum(1.0, c_real + nu_real) + EPS)
                            )
                        ),
                        "status": gate_status(c, nrm),
                    }
                )
                cells.append(f"{c:>5.2f}{_mark(p1, c, nu):>4}")
            print(f"    {p1:>4.2f}  " + "".join(f"{x:>9}" for x in cells))
        print("")

    n_cells = len(rows)
    widths = np.array([r["width"] for r in rows])
    vacuas = [r for r in rows if r["nu"] >= 1.0 - C0 - EPS]
    degen_accept = [r for r in rows if r["accept_degen"] >= 0.5]
    covers = np.array([r["covers"] for r in rows])
    false_valid = [r for r in rows if r["p1"] >= P1_DEGEN and r["accept_degen"] >= 0.5]
    ambig_accept = [
        r for r in rows if P1_VALID < r["p1"] < P1_DEGEN and r["accept_degen"] >= 0.5
    ]
    false_degen = [r for r in rows if r["p1"] <= P1_VALID and r["reject"] >= 0.5]

    print("    RESUMEN DEL MC")
    print(f"      celdas                              : {n_cells}")
    print(f"      ancho medio del intervalo           : {widths.mean():.4f}")
    print(
        f"      ancho medio | nu=0                   : {widths[[r['nu'] == 0.0 for r in rows]].mean():.4f}"
    )
    print(
        f"      ancho medio | nu>=0.30               : {widths[[r['nu'] >= 0.30 - EPS for r in rows]].mean():.4f}"
    )
    print(
        f"      ancho medio | nu<=0.20               : {widths[[r['nu'] <= 0.20 + EPS for r in rows]].mean():.4f}"
    )
    nus_vacuas = sorted(set(r["nu"] for r in vacuas))
    print(
        f"      celdas vacuas (nu >= 1-C0)           : {len(vacuas)}/{n_cells}"
        f"  (nu = {nus_vacuas})"
    )
    print(
        f"      celdas donde el gate ACEPTA y el intervalo admite p1>=0.75 : "
        f"{len(degen_accept)}/{n_cells}"
        f"  (p1>=0.75: {len(false_valid)}; p1 ambiguo: {len(ambig_accept)})"
    )
    print(
        f"      falso-valida (mayoria acepta, p1>=0.75)      : {len(false_valid)}"
        f"  modelos: {sorted(set(r['model'] for r in false_valid))}"
    )
    print(f"      falso-degenerada (mayoria rechaza, p1<=0.65) : {len(false_degen)}")
    print(
        f"      cobertura del intervalo (p1 dentro): {covers.mean():.4f}"
        f"  min={covers.min():.4f}"
    )

    # detalle de las celdas mas informativas
    print("")
    print("    celdas con p1>=0.75 aceptadas y alcance del intervalo:")
    print(
        "      p1    nu   modelo        c     ancho  status(media)   acepta  admite>=0.75"
    )
    for r in rows:
        if r["p1"] >= P1_DEGEN and r["accept_degen"] >= 0.05:
            print(
                f"      {r['p1']:>4.2f}  {r['nu']:>4.2f}  {r['model']:11}"
                f"{r['c']:>6.2f}  {r['width']:>6.2f}  {r['status']:14}"
                f"{1.0 - r['reject']:>7.2f}  {r['accept_degen']:>10.2f}"
            )

    # ── [3b] Frontera de vacuidad: cuanto agujero queda ──────────────────
    print("")
    print("[3b] FRONTERA DE VACUIDAD: p1>=0.75 con k_bounds incluido")
    print("     reach = la particion llega al gate (K en [3,20], nu<=0.50)")
    print("     accept = el gate la acepta (c<=0.70).  fv = acepta | reach")
    print("")
    print("     p1     nu   modelo        c      reach   accept    fv|reach   K medio")
    for p1 in (0.72, 0.75, 0.80):
        for nu in (0.30, 0.35, 0.40):
            for model in ("informative", "uniform"):
                c_r, nu_r, _p, k_r = mc_cell(rng, p1, nu, model, reps)
                reach = (k_r >= 3) & (k_r <= 20) & (nu_r <= 0.50 + EPS)
                acc = reach & (c_r <= C0 + EPS)
                fv = float(acc.sum() / reach.sum()) if reach.sum() else float("nan")
                print(
                    f"     {p1:>4.2f}  {nu:>4.2f}  {model:11}{c_r.mean():>6.2f}"
                    f"  {float(reach.mean()):>7.2f}  {float(acc.mean()):>7.2f}"
                    f"  {fv:>9.2f}  {k_r[reach].mean() if reach.any() else 0:>7.1f}"
                )
    print("")
    print("     -> con nu=0.30 (frontera exacta) algunos degenerados llegan con K>=3")
    print("        y el gate los acepta: el agujero existe y no lo tapa k_bounds.")

    # ── [4] RMSD vs ARI/NMI ────────────────────────────────────────────────
    print("")
    print("[4] RMSD DE PROPORCIONES: no mide acuerdo (contraste con ARI/NMI)")
    try:
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
        )

        have_sk = True
    except ImportError:  # pragma: no cover
        have_sk = False

    def run_case(name, sizes, n, note, perturb=None, other_sizes=None, seed=0):
        rr = np.random.default_rng(seed)
        ref = _labels_from_sizes(sizes)
        if perturb is not None:
            obs = _perturb(ref, perturb, len(sizes), rr)
        elif other_sizes is not None:
            obs = _labels_from_sizes(other_sizes, rr)
        else:
            obs = ref.copy()
        core = _rmsd_core(ref, obs)
        ari = adjusted_rand_score(ref, obs) if have_sk else float("nan")
        nmi = normalized_mutual_info_score(ref, obs) if have_sk else float("nan")
        return {
            "name": name,
            "n": n,
            "rmsd": core["rmsd"],
            "ari": ari,
            "nmi": nmi,
            "note": note,
        }

    print("    (a) corpus grande (n=300): la inversion se reproduce")
    cases = [
        run_case("identidad", [120, 80, 50, 30, 20], 300, "misma particion", seed=11),
        run_case(
            "15% perturbado",
            [120, 80, 50, 30, 20],
            300,
            "acuerdo real alto, perfil distinto",
            perturb=0.15,
            seed=13,
        ),
        run_case(
            "mismo perfil, sin acuerdo",
            [120, 80, 50, 30, 20],
            300,
            "azar con el MISMO vector de tamanos",
            other_sizes=[120, 80, 50, 30, 20],
            seed=12,
        ),
        run_case(
            "perfil distinto, sin acuerdo",
            [120, 80, 50, 30, 20],
            300,
            "azar con OTRO vector de tamanos",
            other_sizes=[60, 60, 60, 60, 60],
            seed=14,
        ),
    ]
    print("      caso                          n     RMSD     ARI     NMI   veredicto")
    for c in cases:
        ok = c["name"] in ("identidad", "15% perturbado")
        print(
            f"      {c['name']:28}{c['n']:>4}  {c['rmsd']:>7.4f}"
            f"  {c['ari']:>6.3f}  {c['nmi']:>6.3f}   {'[ok]' if ok else '[X]'}"
            f"  {c['note']}"
        )
    _inver = [c for c in cases if c["name"] == "mismo perfil, sin acuerdo"][0]
    _real = [c for c in cases if c["name"] == "15% perturbado"][0]
    print("")
    print(
        f"      -> con el MISMO perfil marginal el RMSD es {_inver['rmsd']:.4f} aunque"
        f" el ARI sea {_inver['ari']:.3f}:"
    )
    print("         el estadistico es funcion de los marginales, no de la particion.")
    print(
        f"      -> [X] RMSD ordena el AZAR ({_inver['rmsd']:.4f}) por encima del"
        f" acuerdo real ({_real['rmsd']:.4f}, ARI {_real['ari']:.3f})."
    )
    print("         Misma inversion que el banco: azar 0.0393 < real 0.0478.")
    print("      -> ARI/NMI si separan: identidad 1/1, acuerdo 0.76, azar 0.01.")

    print("")
    print("    (b) resolucion del banco: n=26 UCEs")
    small = [
        run_case(
            "mismo perfil, sin acuerdo",
            [7, 5, 4, 4, 3, 2, 1],
            26,
            "azar con el mismo perfil",
            other_sizes=[7, 5, 4, 4, 3, 2, 1],
            seed=21,
        ),
        run_case(
            "1 UCE movida",
            [7, 5, 4, 3, 3, 2, 2],
            26,
            "una sola UCE cambia de clase",
            perturb=1 / 26,
            seed=22,
        ),
        run_case(
            "mismo perfil, perfil sesgado",
            [16, 3, 2, 2, 1, 1, 1],
            26,
            "azar con el mismo perfil (sesgado)",
            other_sizes=[16, 3, 2, 2, 1, 1, 1],
            seed=23,
        ),
    ]
    print("      caso                          n     RMSD     ARI     NMI")
    for c in small:
        print(
            f"      {c['name']:28}{c['n']:>4}  {c['rmsd']:>7.4f}"
            f"  {c['ari']:>6.3f}  {c['nmi']:>6.3f}"
        )
    print("      -> con n=26 la granularidad de cada proporcion es 1/26 = 0.0385:")
    print("         el 0.0478 y el 0.0393 del banco (diferencia 0.0085) estan POR")
    print("         DEBAJO de una UCE; la inversion no es un efecto sutil de acuerdo,")
    print("         es el ruido de emparejamiento de un corpus de 26 UCEs.")
    print("      -> ademas el valor depende de la forma del perfil (0.08 vs 0.04),")
    print("         luego no es comparable entre corpus de distinto tamano o forma.")

    # ── [5] Contraste con el modulo del repo (opcional) ────────────────────
    if a.check_module:
        print("")
        print("[5] CONTRASTE CON reii.method_selection")
        try:
            from reii.method_selection import MethodMetrics, select_method  # noqa: F401

            has_corpus = hasattr(MethodMetrics("C", 3, 0.5, 0.5), "dominance_corpus")
            print(f"      dominance_corpus implementado: {has_corpus}")
            print(
                f"      gate_status en Selection     : "
                f"{'gate_status' in select_method([]).__dataclass_fields__}"
            )
            if not has_corpus:
                print("      [!] diff no aplicado todavia (ver GATE_INTERVALS.md)")
        except Exception as exc:  # pragma: no cover
            print(f"      [!] no se pudo importar el modulo: {exc}")

    # ── Veredictos ─────────────────────────────────────────────────────────
    print("")
    hr()
    print("VEREDICTOS")
    print(
        f"  [ok] Identificabilidad: p1 in [c, min(1,c+nu)]; ancho medio "
        f"{widths.mean():.3f} (nu<=0.20: "
        f"{widths[[r['nu'] <= 0.20 + EPS for r in rows]].mean():.3f})."
    )
    print(f"       El gate sobre c es cota LIBERAL: certifica rechazo, no aceptacion.")
    print(
        f"  [X] Vacuidad exacta desde nu >= {1.0 - C0:.2f}: "
        f"{len(vacuas)}/{n_cells} celdas ({100.0 * len(vacuas) / n_cells:.0f}%)."
    )
    print(
        f"  [!] {len(degen_accept)}/{n_cells} celdas: el gate acepta y el intervalo "
        f"todavia admite p1>=0.75."
    )
    if have_sk:
        r0 = [c for c in cases if "sin acuerdo" in c["name"] and c["n"] == 300][0]
        print(
            f"  [X] RMSD: mismo perfil + azar -> RMSD={r0['rmsd']:.4f} con "
            f"ARI={r0['ari']:.3f}; reportar ARI/NMI para acuerdo."
        )
    print("  -> RMSD solo como descriptor de perfil, con referencia nombrada.")
    hr()
    return 0


if __name__ == "__main__":
    sys.exit(main())
