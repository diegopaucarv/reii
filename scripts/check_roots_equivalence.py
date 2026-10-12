"""Verifica, con los modelos REALES de Stanza, que los sujetos raíz «sin coref + por lotes»
coinciden con el método original (pipeline completo, una llamada por texto) y mide el tiempo.

Uso (desde la raíz del repo; con PYTHONPATH=src si el paquete no está instalado):

    python scripts/check_roots_equivalence.py --json data/workflow_data.json --n 200
    python scripts/check_roots_equivalence.py --txt frases.txt --n 100 --batch-size 32
    python scripts/check_roots_equivalence.py              # UCEs de PostgreSQL (REII_DATABASE_URL)
    python scripts/check_roots_equivalence.py --cpu        # fuerza CPU

Compara, sobre una muestra de textos (los últimos 500 caracteres de cada UCE, como en COREF):
  ref          pipeline completo (con coref), una llamada por texto   <- comportamiento anterior
  lite_single  sin coref, una llamada por texto
  lite_batch   sin coref, por lotes de --batch-size textos
Código de salida 1 si algún texto produce sujetos distintos al de referencia.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from reii.seg.root_cache import (
    ROOT_PROCESSORS,
    StanzaRootExtractor,
    root_context,
)

FULL_PROCESSORS = "tokenize,pos,lemma,depparse,constituency,coref"


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _measure_memory() -> Dict[str, Any]:
    """RSS y VRAM actuales (defensivo: psutil/torch pueden no estar)."""
    mem: Dict[str, Any] = {}
    try:
        import psutil

        mem["rss_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import resource

            # ru_maxrss: KB en Linux, bytes en macOS.
            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mem["rss_mb"] = (
                kb / 1024.0 if sys.platform != "darwin" else kb / (1024 * 1024)
            )
        except Exception:
            mem["rss_mb"] = None
    try:
        import torch

        if torch.cuda.is_available():
            mem["vram_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            mem["vram_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
    except Exception:
        pass
    return mem


def compare(
    pipe, texts: Sequence[str], batch_size: int = 32, max_examples: int = 10
) -> Dict[str, Any]:
    """Compara los tres métodos sobre ``texts`` con un pipeline ya cargado (real o falso)."""
    texts = list(texts)
    ref_ex = StanzaRootExtractor(lambda: pipe, skip_coref=False)
    lite = StanzaRootExtractor(lambda: pipe, skip_coref=True)
    for t in texts[:2]:  # calentamiento: la primera llamada paga la inicialización
        ref_ex.single(t)
        lite.single(t)

    ref, t_ref = _timed(lambda: [ref_ex.single(t) for t in texts])
    one, t_one = _timed(lambda: [lite.single(t) for t in texts])

    def _batched() -> List[List[str]]:
        out: List[List[str]] = []
        for i in range(0, len(texts), max(1, batch_size)):
            out.extend(lite.batch(texts[i : i + max(1, batch_size)]))
        return out

    bat, t_bat = _timed(_batched)

    def _diff(other: List[List[str]]) -> List[Dict[str, Any]]:
        bad = [i for i, (a, b) in enumerate(zip(ref, other)) if set(a) != set(b)]
        return [
            {
                "i": i,
                "texto": texts[i][-80:],
                "ref": sorted(ref[i]),
                "otro": sorted(other[i]),
            }
            for i in bad
        ]

    d_one, d_bat = _diff(one), _diff(bat)
    n = max(1, len(texts))
    return {
        "n": len(texts),
        "batch_size": batch_size,
        "skip_coref_effective": lite.skip_coref,  # False: esta versión de Stanza no admite processors=
        "seconds": {"ref": t_ref, "lite_single": t_one, "lite_batch": t_bat},
        "ms_per_text": {
            "ref": 1e3 * t_ref / n,
            "lite_single": 1e3 * t_one / n,
            "lite_batch": 1e3 * t_bat / n,
        },
        "speedup_vs_ref": {
            "lite_single": t_ref / t_one if t_one else float("inf"),
            "lite_batch": t_ref / t_bat if t_bat else float("inf"),
        },
        "n_mismatch": {"lite_single": len(d_one), "lite_batch": len(d_bat)},
        "examples": {
            "lite_single": d_one[:max_examples],
            "lite_batch": d_bat[:max_examples],
        },
        "n_empty_ref": sum(1 for r in ref if not r),
        "memory": {"before": _measure_memory(), "after": _measure_memory()},
    }


def render(rep: Dict[str, Any]) -> str:
    lines = [
        f"Textos: {rep['n']} · lote: {rep['batch_size']} · textos sin sujetos (ref): {rep['n_empty_ref']}"
    ]
    if not rep["skip_coref_effective"]:
        lines.append(
            "AVISO: esta versión de Stanza no admite `processors=`; «lite» corrió con el pipeline completo."
        )
    lines.append(
        f"{'método':<13}{'seg':>9}{'ms/texto':>11}{'×vs ref':>9}{'difieren':>10}"
    )
    lines.append(
        f"{'ref':<13}{rep['seconds']['ref']:>9.1f}{rep['ms_per_text']['ref']:>11.1f}{'1.0':>9}{'—':>10}"
    )
    for k in ("lite_single", "lite_batch"):
        lines.append(
            f"{k:<13}{rep['seconds'][k]:>9.1f}{rep['ms_per_text'][k]:>11.1f}"
            f"{rep['speedup_vs_ref'][k]:>9.1f}{rep['n_mismatch'][k]:>10}"
        )
    for k in ("lite_single", "lite_batch"):
        for ex in rep["examples"][k]:
            lines.append(
                f"  [{k}] #{ex['i']} …{ex['texto']!r}\n      ref={ex['ref']}\n      otro={ex['otro']}"
            )
    mem = rep.get("memory", {})
    if mem:

        def _fmt(m: Dict[str, Any]) -> str:
            parts = []
            if m.get("rss_mb") is not None:
                parts.append(f"RSS {m['rss_mb']:.0f} MB")
            if m.get("vram_allocated_mb") is not None:
                parts.append(f"VRAM {m['vram_allocated_mb']:.0f} MB")
            return " · ".join(parts) if parts else "n/d"

        lines.append(
            f"Memoria: antes [{_fmt(mem.get('before', {}))}] · después [{_fmt(mem.get('after', {}))}]"
        )
    ok = not any(rep["n_mismatch"].values())
    lines.append(
        "RESULTADO: "
        + (
            "IGUALES — se puede usar sin coref + lotes."
            if ok
            else "HAY DIFERENCIAS — no activar todavía (ver ejemplos)."
        )
    )
    return "\n".join(lines)


def exit_code(rep: Dict[str, Any]) -> int:
    return 0 if not any(rep["n_mismatch"].values()) else 1


def load_texts(args) -> List[str]:
    if args.txt:
        with open(args.txt, encoding="utf-8") as f:
            raw = [ln.strip() for ln in f if ln.strip()]
    else:
        from reii.quality import migrate_legacy as ml

        legacy = (
            ml.load_legacy_json(args.json)
            if args.json
            else ml.load_legacy_postgres(args.dsn)
        )
        raw = [u.get("texto") or "" for u in legacy["uces"]]
    ctx = list(dict.fromkeys(root_context(t) for t in raw if t and t.strip()))
    random.Random(args.seed).shuffle(ctx)
    return ctx[: args.n]


def _patch_stanza_coref() -> None:
    """Aplica el mismo monkeypatch de ``plateau_epochs`` que gramatical_analyzer.py.

    Sin esto, el pipeline completo (con coref) no carga en esta versión de Stanza
    (``Config.__init__() missing plateau_epochs``) y la comparación full-vs-lite
    sería vacía. El parche es idempotente.
    """
    try:
        from stanza.models.coref.config import Config as _StanzaCorefCfg
    except ImportError:
        return
    if hasattr(_StanzaCorefCfg, "_coref_patched_flag"):
        return
    _StanzaCorefCfg._coref_original_init = _StanzaCorefCfg.__init__

    def _coref_patched_init(self, *args, **kwargs):
        kwargs.setdefault("plateau_epochs", 10)
        _StanzaCorefCfg._coref_original_init(self, *args, **kwargs)

    _StanzaCorefCfg.__init__ = _coref_patched_init
    _StanzaCorefCfg._coref_patched_flag = True


def build_pipeline(lang: str, cpu: bool):
    import stanza

    try:
        import torch

        use_gpu = torch.cuda.is_available() and not cpu
    except ImportError:
        use_gpu = False

    # El pipeline completo (con coref) es la referencia. Aplicamos el parche de
    # plateau_epochs (igual que gramatical_analyzer.py) para que cargue de verdad;
    # si aun así falla, abortamos en vez de comparar lite contra lite.
    _patch_stanza_coref()
    try:
        stanza.download(lang, processors=FULL_PROCESSORS, verbose=False)
        print(
            f"Cargando Stanza ({'GPU' if use_gpu else 'CPU'}, processors={FULL_PROCESSORS})…"
        )
        return stanza.Pipeline(
            lang, processors=FULL_PROCESSORS, use_gpu=use_gpu, verbose=False
        )
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el pipeline completo (con coref): {e}. "
            "La comparación full-vs-lite requiere el modelo de coref."
        ) from e


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--json", help="workflow_data.json (clave «uces»)")
    src.add_argument("--txt", help="archivo de texto, un texto por línea")
    ap.add_argument(
        "--dsn", default=None, help="PostgreSQL (por defecto REII_DATABASE_URL)"
    )
    ap.add_argument("--n", type=int, default=200, help="textos de la muestra")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lang", default="es")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    texts = load_texts(a)
    if not texts:
        print("no hay textos para comparar", file=sys.stderr)
        return 2
    print(f"{len(texts)} textos · procesadores lite: {ROOT_PROCESSORS}")
    rep = compare(build_pipeline(a.lang, a.cpu), texts, a.batch_size)
    print(render(rep))
    return exit_code(rep)


if __name__ == "__main__":
    raise SystemExit(main())
