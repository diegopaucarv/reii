"""Diagnóstico temporal: ¿`processors=` funciona en Stanza 1.11.1 con pipeline coref?"""

import sys
import time
import traceback

sys.path.insert(0, "src")

# Mismo parche plateau_epochs que segmentador.py / gramatical_analyzer.py
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

ROOT_PROCESSORS = "tokenize,mwt,pos,lemma,depparse,constituency"
FULL = "tokenize,pos,lemma,depparse,constituency,coref"

print("Cargando pipeline completo (con coref)...", flush=True)
t0 = time.time()
pipe = stanza.Pipeline("es", processors=FULL, use_gpu=False, verbose=False)
print(f"Pipeline cargado en {time.time() - t0:.1f}s", flush=True)

texto = (
    "El médico residente revisó al paciente y luego escribió la receta en el sistema."
)


def probe(nombre, fn):
    t0 = time.time()
    try:
        out = fn()
        print(
            f"[OK]   {nombre}: {time.time() - t0:.2f}s -> {type(out).__name__}",
            flush=True,
        )
        return out
    except Exception as e:
        print(f"[FAIL] {nombre}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return None


# 1. Llamada individual con processors=
probe("pipe(texto, processors=ROOT)", lambda: pipe(texto, processors=ROOT_PROCESSORS))
# 2. bulk_process con processors=
probe(
    "bulk_process([texto], processors=ROOT)",
    lambda: pipe.bulk_process([texto], processors=ROOT_PROCESSORS),
)
# 3. Llamada individual SIN processors (pipeline completo)
probe("pipe(texto) completo", lambda: pipe(texto))
# 4. bulk_process SIN processors (pipeline completo)
probe("bulk_process([texto]) completo", lambda: pipe.bulk_process([texto]))

# 5. Timing: 10 textos lite por lotes vs individual
textos = [
    f"El paciente número {i} llegó a la consulta y el médico lo atendió con calma."
    for i in range(10)
]
t0 = time.time()
docs = pipe.bulk_process(textos, processors=ROOT_PROCESSORS)
print(f"bulk lite 10 textos: {time.time() - t0:.2f}s", flush=True)
t0 = time.time()
for t in textos:
    pipe(t, processors=ROOT_PROCESSORS)
print(f"single lite 10 textos: {time.time() - t0:.2f}s", flush=True)
t0 = time.time()
docs = pipe.bulk_process(textos)
print(f"bulk FULL 10 textos: {time.time() - t0:.2f}s", flush=True)
t0 = time.time()
for t in textos:
    pipe(t)
print(f"single FULL 10 textos: {time.time() - t0:.2f}s", flush=True)
print("FIN", flush=True)
