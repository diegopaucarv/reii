"""
Banco de pruebas MOCK del optimizador REII.

Qué es: el TPESampler REAL de Optuna 3.6.2 (multivariado, multiobjetivo, como en
main_workflow.Optimizador) corriendo sobre un paisaje SINTÉTICO que reproduce las
DIMENSIONES, los GATES y la ESTRUCTURA DE COSTOS del pipeline (no sus valores).
Qué NO es: una medición del corpus real. Sirve para comparar estrategias de
búsqueda bajo el mismo presupuesto de tiempo, promediando muchos paisajes.

Modelo de costos (unidades = 1 evaluación wc_only):
  c_wc   = 1      trial etapa 1 (CDH + doble pase WC)
  c_all  = 3      trial etapa 2 sin extracción coref (B+C clasificación, embeddings)
  P      = pase completo de extract_coref para un ctx: P*(ctx+1)/3 unidades
  L      = coste de re-segmentar+lematizar el corpus para un uce_target_size nuevo
"""
from __future__ import annotations
import hashlib, itertools, math, random
from dataclasses import dataclass, field
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.ERROR)
import warnings; warnings.filterwarnings("ignore")

# ───────────────────────── espacio (idéntico a get_search_space) ─────────────
SPACE = {
    "min_forms_uc_1": ("int", 6, 14),
    "forms_gap": ("int", 2, 7),
    "tsj": ("int", 2, 5),
    "min_cluster_size_cdh": [0.05, 0.08, 0.10, 0.15, 0.20],
    "swap_iterations": [1, 2, 5],
    "min_r2_threshold": [0.03, 0.05, 0.08, 0.10],
    "similarity_threshold": [0.25, 0.35, 0.45, 0.55, 0.65],
    "coref_weight": [0.0, 0.1, 0.2, 0.3, 0.4],
    "uc_window_size": [1, 2, 3, 4, 5],
    "coref_context_units_tight": [1, 2, 3],
    "coref_context_units_loose": [3, 4, 5],
    "hdbscan_uce_min_cluster_frac": [0.03, 0.05, 0.08, 0.12, 0.15],
    "hdbscan_uce_min_cluster_frac_loose": [0.02, 0.03, 0.05, 0.08, 0.10],
}
UCE_VALUES = [18, 22, 26, 30]
CTX_PAIRS = [(t, l) for t in (1, 2, 3) for l in (3, 4, 5) if t < l]          # 8, exacto
FRAC_T = SPACE["hdbscan_uce_min_cluster_frac"]
FRAC_L = SPACE["hdbscan_uce_min_cluster_frac_loose"]
FRAC_PAIRS = [(a, b) for a in FRAC_T for b in FRAC_L if a > b]               # 16, exacto

WC_KEYS = ["min_forms_uc_1", "forms_gap", "tsj", "min_cluster_size_cdh",
           "swap_iterations", "min_r2_threshold"]
CO_KEYS = ["similarity_threshold", "coref_weight", "uc_window_size",
           "coref_context_units_tight", "coref_context_units_loose",
           "hdbscan_uce_min_cluster_frac", "hdbscan_uce_min_cluster_frac_loose"]


def _t(k, v):  # posición normalizada [0,1] de un valor
    s = SPACE.get(k)
    if k == "uce_target_size":
        return UCE_VALUES.index(v) / (len(UCE_VALUES) - 1)
    if isinstance(s, tuple):
        return (v - s[1]) / (s[2] - s[1])
    return s.index(v) / (len(s) - 1)


# ───────────────────────────── paisaje sintético ─────────────────────────────
class Landscape:
    """Dos etapas correladas (rho): etapa 1 = (ARI_wc, cov, -K), etapa 2 = (cov, agree, -K)."""

    def __init__(self, seed: int, rho: float = 0.7, with_uce: bool = False,
                 noise: float = 0.02):
        r = np.random.default_rng(seed)
        keys = WC_KEYS + CO_KEYS + (["uce_target_size"] if with_uce else [])
        self.keys, self.with_uce, self.noise, self.seed = keys, with_uce, noise, seed
        self.opt2 = {k: r.uniform(0.05, 0.95) for k in keys}        # óptimo etapa 2
        # óptimo etapa 1 = mezcla con perturbación → correlación rho entre etapas
        self.opt1 = {k: float(np.clip(rho * self.opt2[k] + (1 - rho) * r.uniform(0, 1), 0, 1))
                     for k in keys}
        self.w = {k: r.uniform(0.4, 1.6) for k in keys}              # importancia
        # 2 interacciones
        self.inter = [("min_forms_uc_1", "tsj"), ("similarity_threshold", "coref_weight")]
        self.inter_off = {p: r.uniform(-0.3, 0.3) for p in self.inter}
        self.kbias = r.uniform(-0.1, 0.1)

    def _q(self, x, keys, opt, width=0.33):
        num = den = 0.0
        for k in keys:
            if k not in x: continue
            d = (_t(k, x[k]) - opt[k]) / width
            num += self.w[k] * math.exp(-d * d); den += self.w[k]
        q = num / max(den, 1e-9)
        for (a, b) in self.inter:
            if a in x and b in x and a in keys and b in keys:
                d = (_t(a, x[a]) - _t(b, x[b]) - self.inter_off[(a, b)]) / 0.45
                q *= 0.85 + 0.15 * math.exp(-d * d)
        return q

    def _noise(self, x, tag):  # determinista (el pipeline real lo es con random_state fijo)
        h = hashlib.md5(f"{self.seed}|{tag}|{sorted(x.items())}".encode()).digest()
        return (int.from_bytes(h[:4], "little") / 2**32 - 0.5) * 2 * self.noise * 1.7

    def evaluate(self, x, stage: int, noisy=True):
        """→ (valid:bool, objectives tuple). Los gates replican _evaluate_trial."""
        wc_keys = WC_KEYS + (["uce_target_size"] if self.with_uce else [])
        q_wc = self._q(x, wc_keys, self.opt1 if stage == 1 else self.opt2)
        q_co = self._q(x, CO_KEYS, self.opt2)
        nz = (lambda t: self._noise(x, t)) if noisy else (lambda t: 0.0)
        u = 0.55 * (1 - _t("min_cluster_size_cdh", x["min_cluster_size_cdh"])) \
            + 0.30 * (1 - _t("hdbscan_uce_min_cluster_frac", x["hdbscan_uce_min_cluster_frac"])) \
            + 0.15 * (1 - _t("min_forms_uc_1", x["min_forms_uc_1"])) + self.kbias
        K = 2 + 22 * u + 3 * nz("K")
        if stage == 1:
            cov = 0.06 + 0.62 * q_wc + nz("c1")
            gate = 0.25
            ari = 0.05 + 0.80 * q_wc * (0.75 + 0.25 * q_co) + nz("a1")
            obj = (ari, cov, -K)
        else:
            cov = (0.05 + 0.55 * q_wc) * (0.55 + 0.45 * q_co) + nz("c2")
            gate = 0.10
            agree = 0.03 + 0.85 * (0.45 * q_wc + 0.55 * q_co) + nz("a2")
            obj = (cov, agree, -K)
        valid = (2 <= K <= 20) and cov >= gate
        return valid, obj

    @staticmethod
    def utility(obj2):  # utilidad escalar para MEDIR calidad final (no guía la búsqueda)
        cov, agree, negK = obj2
        return 0.5 * cov + 0.5 * agree - 0.004 * (-negK)

    def best_known(self, n=6000, seed=0):
        rng = random.Random(seed); best = -9
        for _ in range(n):
            x = sample_random(rng, self.with_uce)
            ok, obj = self.evaluate(x, 2, noisy=False)
            if ok: best = max(best, self.utility(obj))
        return best


def sample_random(rng, with_uce):
    x = {}
    for k, s in SPACE.items():
        x[k] = rng.randint(s[1], s[2]) if isinstance(s, tuple) else rng.choice(s)
    if with_uce: x["uce_target_size"] = rng.choice(UCE_VALUES)
    return x


# ───────────────────────────── estrategias ──────────────────────────────────
@dataclass
class Strategy:
    name: str
    stage1_frac: float = 0.30
    n_startup: int = 25
    warm_mode: str = "orig"       # orig: Pareto + 2 copias perturbadas idénticas | topk: k distintos
    warm_k: int = 8
    cache_bug: bool = False       # lookup doc_id str≠int → re-extrae B1 y B2 en cada trial
    feasible: str = "post"        # post (centinela post-hoc) | pairs (categórico exacto) | reject
    early_gate: bool = False      # evaluar gates WC antes de pagar B/C
    uce_mode: str = "none"        # none | joint | decoupled
    P: float = 30.0               # coste pase extract_coref (unidades)
    c_wc: float = 1.0
    c_all: float = 3.0
    L: float = 5.0
    gate_frac: float = 0.35       # parte del coste c_all que se ahorra con early_gate al fallar
    s1_cap: int = 120             # tope de trials de etapa 1 en modo tiempo
    max_total: int = 300          # tope global (optimize_trials=300 en rerun_classification)


def _suggest(trial, S: Strategy, with_uce, fixed_uce=None):
    x = {}
    skip = set()
    if S.feasible == "pairs":
        t, l = trial.suggest_categorical("ctx_pair", CTX_PAIRS)
        x["coref_context_units_tight"], x["coref_context_units_loose"] = t, l
        a, b = trial.suggest_categorical("frac_pair", FRAC_PAIRS)
        x["hdbscan_uce_min_cluster_frac"], x["hdbscan_uce_min_cluster_frac_loose"] = a, b
        skip = {"coref_context_units_tight", "coref_context_units_loose",
                "hdbscan_uce_min_cluster_frac", "hdbscan_uce_min_cluster_frac_loose"}
    for k, s in SPACE.items():
        if k in skip: continue
        x[k] = trial.suggest_int(k, s[1], s[2]) if isinstance(s, tuple) else trial.suggest_categorical(k, s)
    if with_uce:
        x["uce_target_size"] = (fixed_uce if fixed_uce is not None
                                else trial.suggest_categorical("uce_target_size", UCE_VALUES))
    return x


def _perturb_orig(x, seed=42):
    """Réplica de Optimizador._perturb_params: Random(seed) NUEVO en cada llamada →
    las 2 copias 'perturbadas' de un mismo trial son idénticas."""
    rng = random.Random(seed); p = dict(x)
    for k, v in x.items():
        s = SPACE.get(k)
        if isinstance(s, tuple):
            step = 1; p[k] = int(max(s[1], min(s[2], v + rng.choice([-step, step]))))
        elif isinstance(s, list) and v in s:
            i = s.index(v); p[k] = s[max(0, min(len(s) - 1, i + rng.choice([-1, 1])))]
    return p


def _to_params(x, S):
    """Convierte x (dict de valores) a params de Optuna según la parametrización."""
    p = dict(x)
    if S.feasible == "pairs":
        p["ctx_pair"] = (x["coref_context_units_tight"], x["coref_context_units_loose"])
        p["frac_pair"] = (x["hdbscan_uce_min_cluster_frac"], x["hdbscan_uce_min_cluster_frac_loose"])
        for k in ("coref_context_units_tight", "coref_context_units_loose",
                  "hdbscan_uce_min_cluster_frac", "hdbscan_uce_min_cluster_frac_loose"):
            p.pop(k)
    return p


def _feasible_pair(x):
    return (x["coref_context_units_tight"] < x["coref_context_units_loose"] and
            x["hdbscan_uce_min_cluster_frac"] > x["hdbscan_uce_min_cluster_frac_loose"])


SENT = (-2.0, -1.0, 0.0)


class Run:
    def __init__(self, S: Strategy, L: Landscape, budget: float, seed: int, max_trials=10_000,
                 n_trials=None):
        self.S, self.L, self.budget, self.seed = S, L, budget, seed
        self.cost = 0.0; self.n_eval = 0
        self.chain_cache = set()        # (uce, ctx) ya extraídos
        self.seg_cache = set()          # uce_target_size ya segmentados
        self.n_trials = n_trials; self.max_trials = max_trials
        self.fixed_uce = None if S.uce_mode != "decoupled" else None
        self.valid_s2 = []              # (x, obs_obj)

    # coste de acceso a cadenas coref para un trial 'all'
    def _coref_cost(self, x):
        S = self.S; uce = x.get("uce_target_size", 26); c = 0.0
        for ctx in (x["coref_context_units_tight"], x["coref_context_units_loose"]):
            if S.cache_bug or (uce, ctx) not in self.chain_cache:
                c += S.P * (ctx + 1) / 3.0
                self.chain_cache.add((uce, ctx))
        return c

    def _seg_cost(self, x):
        u = x.get("uce_target_size")
        if u is None or u in self.seg_cache: return 0.0
        self.seg_cache.add(u); return self.S.L

    def _eval(self, x, stage):
        S, Lc = self.S, self.L
        if not _feasible_pair(x):               # gate previo (sin coste de evaluación)
            return SENT, 0.0, False
        cost = self._seg_cost(x)
        if stage == 1:
            cost += S.c_wc
            ok, obj = Lc.evaluate(x, 1)
        else:
            ok2, obj2 = Lc.evaluate(x, 2)
            cost += self._coref_cost(x)
            ok1, _ = Lc.evaluate(x, 1)          # gate WC barato (para early-gate)
            body = S.c_all
            if S.early_gate and not ok2:
                body = S.c_all * (1 - S.gate_frac)
            cost += body
            ok, obj = ok2, obj2
        return (obj if ok else SENT), cost, ok

    def _stage(self, stage, n_budget, warm, uce_fixed=None):
        S, Lc = self.S, self.L
        study = optuna.create_study(
            directions=["maximize", "maximize", "minimize"],
            sampler=TPESampler(seed=self.seed, n_startup_trials=S.n_startup, multivariate=True))
        for w in warm:
            study.enqueue_trial(_to_params(w, S))
        spent = 0.0; n = 0; attempts = 0
        while spent < n_budget and self.cost < self.budget and n < self.max_trials and self.n_eval < self.S.max_total:
            if self.n_trials is not None and self.n_eval >= self.n_trials: break
            trial = study.ask()
            x = _suggest(trial, S, Lc.with_uce,
                         fixed_uce=uce_fixed if S.uce_mode == "decoupled" and stage == 2 else None)
            if S.uce_mode == "decoupled" and stage == 1 and Lc.with_uce:
                pass  # uce sigue libre en etapa 1 (barata)
            if S.feasible == "reject" and not _feasible_pair(x):
                study.tell(trial, state=TrialState.FAIL); attempts += 1
                if attempts > 400: break
                continue
            vals, c, ok = self._eval(x, stage)
            study.tell(trial, vals)
            spent += c; self.cost += c; n += 1; self.n_eval += 1
            if stage == 2 and ok: self.valid_s2.append((x, vals))
            if stage == 1 and ok: pass
        return study

    def run(self):
        S, Lc = self.S, self.L
        # ── presupuesto por etapa en COSTE (no en trials) ──
        total = self.budget
        b1 = min(total * S.stage1_frac, S.s1_cap * S.c_wc) if self.n_trials is None else float("inf")
        uce_fixed = None
        if self.n_trials is not None:     # modo "n trials" (baseline): 30/70 por número
            n1 = max(20, int(self.n_trials * S.stage1_frac))
            old = self.n_trials; self.n_trials = n1
            st1 = self._stage(1, float("inf"), [])
            self.n_trials = old
        else:
            st1 = self._stage(1, b1, [])
        pareto = [t for t in st1.best_trials if tuple(t.values) != SENT]
        # reconstruir x de cada trial pareto (con el espacio completo)
        warm = [self._x_from_trial(t) for t in pareto]
        if S.uce_mode == "decoupled" and Lc.with_uce and pareto:
            # elegir uce por mayoría ponderada en el frente de etapa 1 (barato, ya medido)
            us = [w["uce_target_size"] for w in warm]
            uce_fixed = max(set(us), key=us.count)
            warm = [dict(w, uce_target_size=uce_fixed) for w in warm]
        if S.warm_mode == "orig":
            enq = []
            for w in warm:
                enq += [w, _perturb_orig(w), _perturb_orig(w)]
        else:
            enq = self._diverse_topk(warm, S.warm_k)
        if S.feasible != "post":
            enq = [e for e in enq if _feasible_pair(e)]
        st2_budget = float("inf")
        self._stage(2, st2_budget, enq, uce_fixed=uce_fixed)
        return self.finish()

    def _x_from_trial(self, t):
        p = dict(t.params); S = self.S
        if S.feasible == "pairs":
            p["coref_context_units_tight"], p["coref_context_units_loose"] = p.pop("ctx_pair")
            p["hdbscan_uce_min_cluster_frac"], p["hdbscan_uce_min_cluster_frac_loose"] = p.pop("frac_pair")
        return p

    def _diverse_topk(self, warm, k):
        if len(warm) <= k: return warm
        keys = WC_KEYS
        pts = [np.array([_t(q, w[q]) for q in keys]) for w in warm]
        chosen = [0]
        while len(chosen) < k:                      # farthest-point sampling
            d = [min(np.linalg.norm(pts[i] - pts[j]) for j in chosen) for i in range(len(pts))]
            chosen.append(int(np.argmax(d)))
        return [warm[i] for i in chosen]

    def finish(self):
        v = self.valid_s2
        if not v: return 0.0, self.cost, self.n_eval, 0
        arr = np.array([o for _, o in v], float)
        mn, mx = arr.min(0), arr.max(0); rng = np.where(mx > mn, mx - mn, 1.0)
        # knee sobre el frente observado (mismo criterio que optimize_preference="knee")
        nd = _nondominated(arr)
        norm = (arr[nd] - mn) / rng
        i = nd[int(np.argmin(np.linalg.norm(norm - np.ones(3), axis=1)))]
        x = v[i][0]
        ok, obj = self.L.evaluate(x, 2, noisy=False)
        u = self.L.utility(obj) if ok else 0.0
        return u, self.cost, self.n_eval, len(v)


def _nondominated(A):
    n = len(A); keep = []
    for i in range(n):
        a = A[i]; dom = False
        for j in range(n):
            if i == j: continue
            b = A[j]
            if (b[0] >= a[0] and b[1] >= a[1] and b[2] >= a[2]) and (b > a).any():
                dom = True; break
        if not dom: keep.append(i)
    return keep or list(range(n))

import sys, json
import sys, json, itertools, numpy as np

def strategies(P):
    base = dict(P=P)
    return {
     "B0 actual (bug cache)":      Strategy("B0", cache_bug=True, **base),
     "S1 fix lookup":              Strategy("S1", **base),
     "S2 +factible exacto (pares)":Strategy("S2", feasible="pairs", **base),
     "S2r +rechazo (FAIL)":        Strategy("S2r", feasible="reject", **base),
     "S3 +presupuesto/warm/gate":  Strategy("S3", feasible="pairs", early_gate=True,
                                    stage1_frac=0.5, n_startup=12, warm_mode="topk", warm_k=8, **base),
    }

def run_all(P, rho, seeds, with_uce=False, extra=None):
    out = {}
    for sd in seeds:
        Ls = Landscape(sd, rho=rho, with_uce=with_uce)
        bk = Ls.best_known(2500)
        strat = strategies(P)
        if extra: strat.update(extra(P))
        # Fase A: 90 trials iguales
        for name, S in strat.items():
            u, c, n, v = Run(S, Ls, budget=1e12, seed=sd, n_trials=90).run()
            out.setdefault(("A90", name), []).append((u/bk, c, n, v))
        # Fase B: mismo TIEMPO que cuesta hoy un 90-trials (con bug) y que cuesta con fix
        T_bug = out[("A90", "B0 actual (bug cache)")][-1][1]
        T_fix = out[("A90", "S1 fix lookup")][-1][1]
        for tag, T in (("T=coste90_actual", T_bug), ("T=coste90_con_fix", T_fix)):
            for name, S in strat.items():
                u, c, n, v = Run(S, Ls, budget=T, seed=sd).run()
                out.setdefault((tag, name), []).append((u/bk, c, n, v))
    return out

def summarize(out, title):
    print("\n" + title)
    for tag in ("A90", "T=coste90_actual", "T=coste90_con_fix"):
        print(f"  [{tag}]")
        ref = np.array([r[0] for r in out[(tag, "S1 fix lookup")]])
        for (t, name), rows in out.items():
            if t != tag: continue
            a = np.array(rows); u = a[:, 0]; d = u - ref
            ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
            print(f"    {name:32s} util/best={u.mean():.3f}  Δvs S1={d.mean():+.3f}±{ci:.3f}"
                  f"  trials={a[:,2].mean():5.0f} coste={a[:,1].mean():7.0f} valid2={a[:,3].mean():5.1f}")


# ════════════════════════════════════════════════════════════════════════════
#  REPORTE: verificación del patch + tests + benchmark del optimizador
# ════════════════════════════════════════════════════════════════════════════
import argparse
import datetime as _dt
import os
import re
import shutil
import subprocess
import tempfile
import time


def _run(cmd, cwd=None, timeout=600):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        return -1, f"TIMEOUT tras {timeout}s: {e}"


def check_patch(repo, base, patch):
    """Clona el repo en un directorio temporal, checkout del commit base y aplica el patch."""
    lines = ["## 1. Verificación del patch", ""]
    tmp = tempfile.mkdtemp(prefix="reii_patchcheck_")
    try:
        rc, out = _run(["git", "clone", "-q", repo, tmp])
        if rc != 0:
            lines.append(f"- ❌ No se pudo clonar `{repo}`: {out.strip()[:300]}")
            return lines, False
        rc, out = _run(["git", "checkout", "-q", base], cwd=tmp)
        if rc != 0:
            lines.append(f"- ❌ No existe el commit base `{base}`")
            return lines, False
        rc, out = _run(["git", "apply", "--check", os.path.abspath(patch)], cwd=tmp)
        if rc != 0:
            lines.append(f"- ❌ El patch NO aplica sobre `{base}`:\n\n```\n{out.strip()[:800]}\n```")
            return lines, False
        rc, out = _run(["git", "apply", os.path.abspath(patch)], cwd=tmp)
        lines.append(f"- ✅ El patch aplica limpio sobre `{base}`")
        rc2, out2 = _run([sys.executable, "-c",
                          "import ast,sys;[ast.parse(open(f,encoding='utf-8').read(),f) "
                          "for f in sys.argv[1:]];print('ok')",
                          os.path.join(tmp, "src/reii/main_workflow.py"),
                          os.path.join(tmp, "src/reii/coref_cache.py")])
        lines.append(f"- {'✅' if rc2 == 0 else '❌'} Sintaxis Python de los archivos modificados "
                     f"({'OK' if rc2 == 0 else out2.strip()[:300]})")
        return lines, rc == 0 and rc2 == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_unit_tests(repo, tests):
    lines = ["", "## 2. Tests unitarios", ""]
    ok_all = True
    lines.append(f"Directorio: `{os.path.abspath(repo)}`")
    rc, out = _run(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
    if rc == 0:
        lines.append(f"Commit: `{out.strip()}`")
    rc, out = _run(["git", "status", "--short"], cwd=repo)
    if rc == 0:
        lines.append(f"Archivos modificados sin commit: {len([l for l in out.splitlines() if l.strip()])}")
    lines += ["", "| Archivo | Resultado |", "|---|---|"]
    details = []
    for t in tests:
        path = t if os.path.isabs(t) else os.path.join(repo, t)
        if not os.path.exists(path):
            lines.append(f"| `{t}` | ⚠️ no existe |"); ok_all = False; continue
        rc, out = _run([sys.executable, "-m", "pytest", path, "-q", "--tb=short", "-p", "no:cacheprovider"],
                       cwd=repo, timeout=900)
        summ = [l.strip() for l in out.splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
        status = summ[-1] if summ else out.strip()[-160:]
        mark = "✅" if rc == 0 else "❌"
        lines.append(f"| `{t}` | {mark} {status} |")
        if rc != 0:
            ok_all = False
            details += ["", f"### Detalle de fallos: `{t}`", "", "```", out.strip()[-3500:], "```"]
    return lines + details, ok_all


def run_benchmark(seeds, log, scenarios=None, on_scenario=None):
    lines = ["", "## 3. Benchmark (simulación mock, TPE real de Optuna)", ""]
    lines.append(f"Paisajes sintéticos por escenario: **{seeds}**. Δ = utilidad relativa al óptimo conocido "
                 "menos la de la estrategia de referencia (S1 = solo fix de lookup). IC95 aprox. = 1.96·sd/√n.")
    lines.append("")
    scenarios = scenarios or [(30, 0.7), (100, 0.7), (30, 0.3)]
    all_results = {}
    for P, rho in scenarios:
        t0 = time.time()
        out = run_all(P, rho, list(range(seeds)))
        all_results[f"P={P},rho={rho}"] = {f"{k[0]}|{k[1]}": v for k, v in out.items()}
        lines.append(f"### Escenario P={P} (coste pase coref por ctx), rho={rho} (correlación etapa 1↔2)")
        lines.append("")
        for tag, title in (("A90", "A. Mismo nº de trials (90)"),
                           ("T=coste90_actual", "B. Mismo tiempo que el 90-trials actual (con bug)"),
                           ("T=coste90_con_fix", "C. Mismo tiempo que el 90-trials con el fix")):
            lines.append(f"**{title}**")
            lines.append("")
            lines.append("| Estrategia | utilidad/óptimo | Δ vs S1 (IC95) | trials | coste | válidos etapa 2 |")
            lines.append("|---|---|---|---|---|---|")
            ref = np.array([r[0] for r in out[(tag, "S1 fix lookup")]])
            for (t, name), rows in out.items():
                if t != tag: continue
                a = np.array(rows); u = a[:, 0]; d = u - ref
                ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                lines.append(f"| {name} | {u.mean():.3f} | {d.mean():+.3f} ± {ci:.3f} | "
                             f"{a[:,2].mean():.0f} | {a[:,1].mean():.0f} | {a[:,3].mean():.1f} |")
            lines.append("")
        log(f"  escenario P={P} rho={rho} listo en {time.time()-t0:.0f}s")
        if on_scenario: on_scenario(lines)
    return lines, all_results


def main():
    ap = argparse.ArgumentParser(
        description="Reporte REII: tests unitarios + benchmark mock del optimizador (uso directo sobre tu repo)")
    ap.add_argument("--workdir", default=".", help="Raíz del repo REII (default: directorio actual)")
    ap.add_argument("--tests", nargs="+",
                    default=["tests/test_optimizer_budget_bugs.py", "tests/test_segmentador_compact.py", "tests/test_checkpoint_hardening.py"],
                    help="Archivos de test a ejecutar (rutas relativas a --workdir)")
    ap.add_argument("--seeds", type=int, default=8, help="Nº de paisajes sintéticos por escenario")
    ap.add_argument("--skip-bench", action="store_true", help="Solo tests, sin benchmark")
    ap.add_argument("--skip-tests", action="store_true", help="Solo benchmark, sin tests")
    ap.add_argument("--scenarios", default="30:0.7,100:0.7,30:0.3",
                    help="Lista P:rho separada por comas (coste pase coref : correlación etapa1-2)")
    ap.add_argument("--out", default="reii_optimizer_report.md")
    # verificación opcional del patch (clona en un tmp; NO se usa por defecto)
    ap.add_argument("--check-patch", default=None, help="Ruta a un .patch para verificar que aplica sobre --base")
    ap.add_argument("--repo", default=None, help="URL/ruta del repo a clonar (solo con --check-patch)")
    ap.add_argument("--base", default="9970b8e", help="Commit base (solo con --check-patch)")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    head = ["# Reporte REII — optimizador y segmentador",
            f"Generado: {_dt.datetime.now().isoformat(timespec='seconds')}",
            f"Python {sys.version.split()[0]} · numpy {np.__version__} · optuna {optuna.__version__}", ""]
    body, ok_all = [], True

    if args.check_patch:
        log("→ verificando patch")
        l, ok = check_patch(args.repo or args.workdir, args.base, args.check_patch); body += l; ok_all &= ok

    if not args.skip_tests:
        log("→ tests unitarios")
        l, ok = run_unit_tests(args.workdir, args.tests); body += l; ok_all &= ok

    results = {}
    if not args.skip_bench:
        sc = [tuple(float(v) for v in x.split(":")) for x in args.scenarios.split(",")]
        log(f"→ benchmark ({args.seeds} paisajes × {len(sc)} escenarios; ~65 s por paisaje y escenario)")
        l, results = run_benchmark(args.seeds, log, scenarios=[(int(p), r) for p, r in sc],
                                   on_scenario=lambda bl: open(args.out, "w", encoding="utf-8").write(
                                       "\n".join(head + body + bl) + "\n\n> (parcial: benchmark en curso)\n"))
        body += l

    body += ["", "## 4. Resumen", "",
             "- Tests: " + ("✅ todo OK" if ok_all else "❌ hay fallos, ver sección 2"),
             "- Benchmark: Δ vs S1 con su IC95; si el IC cruza 0 no hay evidencia de diferencia."]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + body) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump({"tests_ok": bool(ok_all), "benchmark": results}, fh,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1)
    log(f"→ reporte escrito en {args.out} (+ {os.path.splitext(args.out)[0]}.json)")


if __name__ == "__main__":
    main()
