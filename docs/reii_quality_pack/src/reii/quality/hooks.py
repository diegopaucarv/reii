"""Integración con el workflow: guarda una corrida de calidad tras ``_save_all_uces``.

Nunca lanza excepciones hacia el pipeline: si algo falla, registra un warning y
devuelve None (la corrida de calidad es un complemento, no parte del cálculo).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from reii.quality import calibration as cal
from reii.quality import consensus as cons
from reii.quality import metrics as M
from reii.quality import service
from reii.quality.store import QualityStore

logger = logging.getLogger(__name__)

FLAG_ATTRS = {
    "wc": "stable_wc",
    "sim": "stable_sim",
    "emb": "stable_emb",
    "cross_ab": "stable_cross_ab",
    "cross_bc": "stable_cross_bc",
    "cross_ac": "stable_cross_ac",
}
_SECRET = re.compile(r"(api_?key|secret|password|dsn|passwd)", re.I)


def redact_config(cfg: Any) -> Dict[str, Any]:
    d = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg or {})
    return {k: ("***" if _SECRET.search(k) else v) for k, v in d.items()}


def code_version() -> str:
    env = os.environ.get("REII_CODE_VERSION")
    if env:
        return env
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _flat(
    uces_por_doc: Iterable[Iterable[Any]], extra: Optional[Iterable[Any]] = None
) -> List[Any]:
    seen, out = set(), []
    for doc in uces_por_doc or []:
        for u in doc:
            if u.id not in seen:
                seen.add(u.id)
                out.append(u)
    for u in extra or []:
        if u.id not in seen:
            seen.add(u.id)
            out.append(u)
    return out


def build_assignments(uces: List[Any]) -> Tuple[pd.DataFrame, List[str]]:
    """DataFrame por UCE con puntaje de consenso. Devuelve (df, comparaciones_disponibles)."""
    ids = [str(u.id) for u in uces]
    stable_sets: Dict[str, List[str]] = {}
    for name, attr in FLAG_ATTRS.items():
        flags = [getattr(u, attr, None) for u in uces]
        if any(f is not None for f in flags):
            stable_sets[name] = [str(u.id) for u, f in zip(uces, flags) if f]
    avail = [n for n, s in stable_sets.items() if s]
    if avail:
        agree = cons.agreement_from_stable_sets(ids, stable_sets, available=avail)
        source = "consenso_" + "+".join(avail)
    else:  # modos sin banderas por comparación: solo is_stable
        agree = pd.DataFrame(index=pd.Index(ids, name="uce_id"))
        agree["n_comparisons"] = 1
        agree["s_acuerdo"] = [
            1.0 if getattr(u, "is_stable", False) else 0.0 for u in uces
        ]
        source = "is_stable"
    rows = []
    for u in uces:
        uid = str(u.id)
        stable = bool(getattr(u, "is_stable", False))
        cid = getattr(u, "cluster_id", None)
        comps = (
            {n: (uid in set(stable_sets[n])) for n in stable_sets}
            if stable_sets
            else {}
        )
        rows.append(
            {
                "uce_id": uid,
                "doc_id": getattr(u, "doc_id", None),
                "cluster_core": cid
                if (stable and cid is not None and cid >= 0)
                else None,
                "s_margen": getattr(u, "projection_margin", None),
                "projected_cluster_id": getattr(u, "projected_cluster_id", None),
                "n_tokens": len(getattr(u, "lemmas", []) or []),
                "n_types": len(set(getattr(u, "lemmas", []) or [])),
                "comps": comps,
                "s_source": source,
            }
        )
    df = pd.DataFrame(rows)
    df = df.merge(
        agree[["n_comparisons", "s_acuerdo"]],
        left_on="uce_id",
        right_index=True,
        how="left",
    )
    return df, avail


def _uce_dict(u: Any) -> Dict[str, Any]:
    return {
        k: getattr(u, k, None)
        for k in (
            "lemmas",
            "stems",
            "bigrams",
            "bigram_stems",
            "trigrams",
            "trigram_stems",
        )
    }


def build_extras(
    wf: Any,
    stable_uces: List[Any],
    all_uces: List[Any],
    vocab: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Indicadores de cobertura (types como unidad) y estabilidad por clase.

    La cobertura se calcula con :func:`M.corpus_coverage`: types en las UCEs
    estables / types del corpus (total y activo post-TSJ). ``all_uces`` aporta el
    vocabulario del corpus (con frecuencias para el filtro TSJ); ``stable_uces``
    aporta el numerador.
    """
    extras: Dict[str, Any] = {"vocab_scope": "uces_estables"}
    cfg = wf.config
    use_stems = getattr(cfg, "stem_backend", "snowball") != "none"
    use_bigrams = getattr(cfg, "use_bigrams", True)
    use_trigrams = getattr(cfg, "use_trigrams", True)
    tsj = int(getattr(cfg, "tsj", 3))
    n_perm = int(getattr(cfg, "null_perm", 200))

    def _terms(u: Any) -> List[str]:
        return list(M.iter_terms(_uce_dict(u), use_stems, use_bigrams, use_trigrams))

    corpus_term_lists = [_terms(u) for u in all_uces]
    stable_term_lists = [_terms(u) for u in stable_uces]

    if corpus_term_lists and any(corpus_term_lists):
        extras["coverage"] = M.corpus_coverage(
            stable_term_lists, corpus_term_lists, tsj
        )
        by_class: Dict[int, List[List[str]]] = {}
        for u in stable_uces:
            cid = getattr(u, "cluster_id", None)
            if cid is None or cid < 0:
                continue
            by_class.setdefault(int(cid), []).append(_terms(u))
        if by_class:
            extras["coverage_by_class"] = M.coverage_by_class(
                by_class, corpus_term_lists, tsj
            )
        extras["null_coverage"] = M.null_coverage(
            stable_term_lists, corpus_term_lists, tsj, n_perm=n_perm
        )
    elif vocab:
        # fallback sin datos de términos: cobertura sobre el vocabulario explícito
        term_lists = [_terms(u) for u in stable_uces]
        extras["coverage"] = M.vocabulary_coverage(term_lists, vocab)

    pw = getattr(getattr(wf, "double_clf", None), "last_pairwise_stability", None) or {}
    ov = pw.get("overlap_aa")
    if ov:
        extras["jaccard_by_class"] = {
            str(k): v for k, v in M.jaccard_from_overlap(np.array(ov)).items()
        }
    for k in (
        "ari_wc",
        "ari_coref",
        "ari_emb",
        "ari_cross_ab",
        "ari_cross_bc",
        "ari_cross_ac",
    ):
        if k in pw:
            extras[k] = pw[k]
    extras["ari"] = pw.get("ari_wc") or pw.get("ari_coref") or pw.get("ari_emb")
    return extras


def save_quality_run(
    wf: Any,
    uces_est_list: List[Any],
    uces_por_doc: List[List[Any]],
    store: Optional[QualityStore] = None,
    label: Optional[str] = None,
    scope: str = "default",
) -> Optional[str]:
    try:
        uces = _flat(uces_por_doc, uces_est_list)
        if not uces:
            return None
        stable_uces = [u for u in uces if getattr(u, "is_stable", False)]
        df, avail = build_assignments(uces)
        cfg_dict = redact_config(wf.config)
        corpus_hash = hashlib.sha1(
            "\n".join(
                sorted(
                    f"{u.id}|{hashlib.sha1((getattr(u, 'texto', '') or '').encode()).hexdigest()[:8]}"
                    for u in uces
                )
            ).encode()
        ).hexdigest()[:16]
        cv = code_version()
        seed = getattr(wf.config, "random_state", None)
        mode = getattr(wf.config, "classification_mode", None)
        canon = json.dumps(cfg_dict, sort_keys=True, default=str)
        run_id = (
            "run-"
            + hashlib.sha1(
                f"{corpus_hash}|{cv}|{seed}|{mode}|{canon}".encode()
            ).hexdigest()[:16]
        )
        vocab = getattr(wf, "_vocab", None) or wf.db.data.get("vocabulario")
        extras = build_extras(wf, stable_uces, uces, vocab)
        extras["comparaciones_disponibles"] = avail

        owns = store is None
        store = store or QualityStore.open()
        try:
            store.save_run(
                {
                    "run_id": run_id,
                    "label": label or f"{mode} · seed {seed}",
                    "kind": "pipeline",
                    "mode": mode,
                    "seed": seed,
                    "corpus_hash": corpus_hash,
                    "code_version": cv,
                    "config": cfg_dict,
                    "summary": {"extras": extras},
                    "n_uces": len(df),
                    "n_uces_total": len(uces),
                }
            )
            store.save_assignments(run_id, df)
            service.evaluate_run(store, run_id, cal.default_profile(), persist=True)
            if store.get_active(scope) is None:
                store.set_active(
                    scope, run_id, "estricto", cal.profile_hash(cal.default_profile())
                )
            store.log(
                run_id, "save_quality_run", {"n_uces": len(df), "comparaciones": avail}
            )
        finally:
            if owns:
                store.close()
        logger.info("quality: corrida %s guardada (%d UCEs)", run_id, len(df))
        return run_id
    except Exception as e:  # pragma: no cover - defensivo
        logger.warning("quality: no se pudo guardar la corrida: %s", e)
        return None
