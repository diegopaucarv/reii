"""Perfiles de niveles (umbrales), calibración isotónica y asignación anidada.

Niveles por defecto (valores propuestos; fijarlos de antemano):
    estricto    : s_acuerdo >= 1.0
    equilibrado : p_cal >= 0.80   (sin calibración: s_acuerdo >= 0.66)
    amplio      : p_cal >= 0.70 (sin calibración: s_acuerdo >= 0.50) + proyectadas
Los niveles están anidados por construcción: estricto ⊆ equilibrado ⊆ amplio.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

DEFAULT_PROFILE: Dict[str, Any] = {
    "name": "default-v1",
    "tiers": [
        {"tier_id": "estricto", "name": "Estricto", "kind": "s_min", "threshold": 1.0,
         "fallback_s": 1.0, "include_projected": False},
        {"tier_id": "equilibrado", "name": "Equilibrado", "kind": "p_min", "threshold": 0.80,
         "fallback_s": 0.66, "include_projected": False},
        {"tier_id": "amplio", "name": "Amplio", "kind": "p_min", "threshold": 0.70,
         "fallback_s": 0.50, "include_projected": True},
    ],
}


def default_profile() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_PROFILE)


def profile_hash(profile: Dict[str, Any]) -> str:
    canon = json.dumps(profile, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]


REGISTERED_HASH = profile_hash(DEFAULT_PROFILE)


def is_exploratory(profile: Dict[str, Any], registered: str = REGISTERED_HASH) -> bool:
    return profile_hash(profile) != registered


# ── Calibración ─────────────────────────────────────────────────────────────


def fit_isotonic(scores, correct, min_n: int = 60) -> Optional[Dict[str, Any]]:
    """Ajusta P(correcto | puntaje). Devuelve None si hay menos de ``min_n`` casos."""
    s = np.asarray(scores, float)
    y = np.asarray(correct, float)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    if len(s) < min_n or len(np.unique(y)) < 2:
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(s, y)
    return {
        "method": "isotonic",
        "x": [float(v) for v in iso.X_thresholds_],
        "y": [float(v) for v in iso.y_thresholds_],
        "n": int(len(s)),
    }


def apply_calibration(cal: Optional[Dict[str, Any]], scores) -> np.ndarray:
    s = np.asarray(scores, float)
    if not cal:
        return np.full(len(s), np.nan)
    return np.interp(s, cal["x"], cal["y"])


def threshold_for_precision(
    scores, correct, target: float, min_n: int = 30
) -> Optional[float]:
    """Mínimo puntaje t tal que la precisión empírica de {s >= t} sea >= target."""
    s = np.asarray(scores, float)
    y = np.asarray(correct, float)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    best = None
    cum = np.cumsum(y)
    for k in range(min_n, len(s) + 1):
        if cum[k - 1] / k >= target:
            best = float(s[k - 1])
    return best


# ── Asignación de niveles ───────────────────────────────────────────────────


def assign_tiers(
    df: pd.DataFrame, profile: Dict[str, Any], calibration: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Añade columnas ``in_<tier_id>`` (anidadas), ``tier_min`` y ``shown_cluster_<tier_id>``.

    df necesita: cluster_core, s_acuerdo, projected_cluster_id (p_cal se recalcula
    si hay calibración).
    """
    out = df.copy()
    s = pd.to_numeric(out["s_acuerdo"], errors="coerce").fillna(0.0).to_numpy()
    core = pd.to_numeric(out["cluster_core"], errors="coerce")
    inductive = (core.notna() & (core >= 0)).to_numpy()
    proj = pd.to_numeric(out.get("projected_cluster_id"), errors="coerce")
    projected = (proj.notna() & (proj >= 0)).to_numpy() & ~inductive

    p = apply_calibration(calibration, s)
    out["p_cal"] = p if calibration else np.nan

    acc_ind = np.zeros(len(out), bool)
    acc_proj = np.zeros(len(out), bool)
    tier_ids: List[str] = []
    shown = np.where(inductive, core, np.where(projected, proj, np.nan))
    for t in profile["tiers"]:
        tid = t["tier_id"]
        tier_ids.append(tid)
        if t["kind"] == "s_min":
            ok = s >= float(t["threshold"]) - 1e-12
        elif calibration is not None:
            ok = p >= float(t["threshold"]) - 1e-12
        else:
            ok = s >= float(t.get("fallback_s", t["threshold"])) - 1e-12
        acc_ind = acc_ind | (inductive & ok)  # anidado: lo ya incluido se conserva
        if t.get("include_projected"):
            acc_proj = acc_proj | projected
        cur = acc_ind | acc_proj
        out[f"in_{tid}"] = cur
        out[f"shown_cluster_{tid}"] = np.where(cur, shown, np.nan)
    tier_min = pd.Series([None] * len(out), index=out.index, dtype=object)
    for tid in reversed(tier_ids):
        tier_min = tier_min.where(~out[f"in_{tid}"], tid)
    out["tier_min"] = tier_min
    return out


def check_nesting(df: pd.DataFrame, tier_ids: List[str]) -> bool:
    """Estricto ⊆ Equilibrado ⊆ Amplio (en el orden dado)."""
    for a, b in zip(tier_ids, tier_ids[1:]):
        if (df[f"in_{a}"] & ~df[f"in_{b}"]).any():
            return False
    return True
