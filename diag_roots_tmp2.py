"""Diagnóstico temporal 2: código REAL de root_cache con pipeline REAL y textos reales."""

import json
import random
import sys
import time

sys.path.insert(0, "src")

# Mismo parche plateau_epochs que segmentador.py
try:
    from stanza.models.coref.config import Config as _StanzaCorefCfg

    if not hasattr(_StanzaCorefCfg, "_coref_patched_flag"):
        _StanzaCorefCfg._coref_original_init = _StanzaCorefCfg.__init__

        def _coref_patched_init(self, *args, **kwargs):
            kwargs.setdefault("plateau_epochs", 10)
            _StanzaCorefCfg._coref_original_init(self, *args, **kwargs)

        _StanzaCorefCfg.__init__ = _coref_patched_init
        _StanzaCorefCfg._coref_patched_flag = True
except Exception as e:
    print("patch skip:", e)

import stanza

from reii.seg.root_cache import (
    ROOT_PROCESSORS,
    RootSubjectCache,
    StanzaRootExtractor,
    root_context,
)

# Textos reales: últimos 500 chars de segmentos de entrevistas
d = json.load(open("data/entrevistas.json", encoding="utf-8"))
ctxs = []
for e in d["entrevistas"]:
    for s in e["segmentos"]["segmentos"]:
        t = s["texto_literal"]
        if t and t.strip():
            ctxs.append(root_context(t))
ctxs = list(dict.fromkeys(ctxs))
random.Random(1).shuffle(ctxs)
ctxs = ctxs[:47]
print(
    f"{len(ctxs)} contextos reales (últimos 500 chars), longitudes: "
    f"min={min(len(c) for c in ctxs)} max={max(len(c) for c in ctxs)}",
    flush=True,
)

print("Cargando pipeline completo (con coref)...", flush=True)
t0 = time.time()
pipe = stanza.Pipeline(
    "es",
    processors="tokenize,pos,lemma,depparse,constituency,coref",
    use_gpu=False,
    verbose=False,
)
print(f"Pipeline cargado en {time.time() - t0:.1f}s", flush=True)

ex = StanzaRootExtractor(lambda: pipe, skip_coref=True)
print(f"skip_coref inicial: {ex.skip_coref}", flush=True)

# 1) batch() con el código real
t0 = time.time()
out = ex.batch(ctxs)
tb = time.time() - t0
print(
    f"batch() real: {tb:.1f}s para {len(ctxs)} textos = {tb / len(ctxs):.2f}s/texto | "
    f"skip_coref tras batch: {ex.skip_coref} | docs: {len(out)}",
    flush=True,
)

# 2) single() con el código real (muestra de 10)
t0 = time.time()
for c in ctxs[:10]:
    ex.single(c)
ts = time.time() - t0
print(
    f"single() real (10 textos): {ts:.1f}s = {ts / 10:.2f}s/texto | "
    f"skip_coref tras single: {ex.skip_coref}",
    flush=True,
)

# 3) RootSubjectCache.prefetch con batch_extractor real
rc = RootSubjectCache(ex.single, ex.batch, batch_size=32, max_batch_failures=3)
t0 = time.time()
n = rc.prefetch(ctxs, label="ctx=2 ")
tp = time.time() - t0
print(
    f"prefetch() real: {tp:.1f}s para {n} textos = {tp / max(n, 1):.2f}s/texto | "
    f"batch activo: {rc._batch is not None} | fallos: {rc._batch_failures} | "
    f"stats: {rc.stats()}",
    flush=True,
)
print("FIN", flush=True)
