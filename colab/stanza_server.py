!pkill -f uvicorn
!pkill -f cloudflare

# ── 0. Fix de entorno: torchao desactualizado rompe la importación de torch ──
# Colab trae torchao 0.10.0 preinstalado, pero el torch reciente (2.9+) exige
# torchao >= 0.16.0. Sin este upgrade, `import stanza` (que importa torch)
# lanza: ImportError: Found an incompatible version of torchao.
!pip install -U "torchao>=0.16.0" -q

import asyncio
import contextlib
import gc
import itertools
import os
import re
import secrets
import shlex
import subprocess

import stanza
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

os.environ["REII_STANZA_FP16"] = "0"         # dejalo en 0 por ahora — ver nota abajo

# ── 1. Configuración ────────────────────────────────────────────────────────────
API_TOKEN = "6b92bb50d646a9a4030ac1adf0a5cc8e"
LANG = "es"
PROCESSORS = "tokenize,mwt,pos,lemma,depparse,constituency,coref"
# Subconjunto para sujetos raíz: sin coref (mismo patrón que root_cache.py).
ROOT_PROCESSORS = "tokenize,mwt,pos,lemma,depparse,constituency"

# ── 1a. Réplicas del pipeline (opción 3) ─────────────────────────────────────
# No existe forma segura de batchear varias ventanas de coref en una sola
# llamada nlp() (ver nota larga junto a /v1/extract_coref — verificado contra
# el código real de Stanza). La única manera de usar más de un "carril" de
# GPU en paralelo es tener VARIAS instancias del pipeline cargadas, cada una
# resolviendo una ventana INDEPENDIENTE a la vez — nunca mezclando texto de
# dos ventanas en el mismo forward pass, solo corriendo N forward passes
# simultáneos de ventanas distintas.
#
# ⚠ Cada réplica es una copia completa del pipeline (~varios GB de VRAM).
# Dos modos:
#   REII_STANZA_N_REPLICAS=auto  (default) → el server decide solo: carga
#     réplicas de a una, corre una inferencia de prueba en cada una (para
#     medir memoria REAL de uso — pesos + activaciones — no solo los pesos
#     cargados) y para apenas el margen libre cae debajo de
#     REII_STANZA_VRAM_SAFETY_MARGIN_GB, o si un CUDA OOM directamente
#     interrumpe la carga. Nunca hace falta leer el log y decidir a mano.
#   REII_STANZA_N_REPLICAS=<entero> → fuerza ese número exacto, sin probing
#     (modo manual, por si alguna vez el auto-sizing da un número con el que
#     no estás cómodo y preferís fijarlo vos).
_N_REPLICAS_ENV = os.environ.get("REII_STANZA_N_REPLICAS", "auto").strip().lower()
AUTO_SIZE_REPLICAS = _N_REPLICAS_ENV == "auto"
MAX_REPLICAS = int(os.environ.get("REII_STANZA_MAX_REPLICAS", "4"))
VRAM_SAFETY_MARGIN_GB = float(os.environ.get("REII_STANZA_VRAM_SAFETY_MARGIN_GB", "1.5"))
# Texto de prueba para forzar un forward pass real (tokenize→coref) al medir
# cada réplica — un pipeline recién cargado sin inferencia todavía no reservó
# los buffers de activación/coref que sí se reservan en producción, así que
# medir solo después de .Pipeline(...) subestimaría la memoria real.
_VRAM_PROBE_TEXT = (
    "Juan llegó temprano a la reunión del equipo. Él explicó con calma el "
    "problema que habían detectado la semana anterior, mientras los demás "
    "escuchaban atentos y tomaban notas sobre lo que Juan estaba planteando."
)

# ── 1b. Precisión reducida (opción 4) ────────────────────────────────────────
# Desactivado por default — es un experimento, no un default seguro. Antes
# de dejarlo prendido en una corrida real: correlo sobre una ventana de
# prueba con REII_STANZA_FP16=1 vs =0 y compará las cadenas de coref
# resultantes (mismo texto, mismo context_units) para confirmar que no se
# pierden/alteran menciones. autocast (no .half() sobre los pesos) porque es
# la forma estándar de mixed precision en inferencia: Stanza/torch deciden
# internamente qué operaciones bajar a fp16 y cuáles mantener en fp32 (p. ej.
# softmax, layer norm), en vez de forzar todo el modelo a fp16 de una.
USE_FP16 = os.environ.get("REII_STANZA_FP16", "0") == "1"


@contextlib.contextmanager
def _maybe_autocast():
    if USE_FP16:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


# ── 1c. Túnel configurable (opción 5) ────────────────────────────────────────
# El límite de ~100s de `cloudflared tunnel --url` (túnel rápido, sin cuenta)
# es la razón de ser de MAX_WINDOW_CHARS/CHUNK_OVERLAP/REMOTE_BATCH en
# segmentador.py. Si migrás a un túnel propio (SSH reverso, ngrok con plan
# pago, Cloudflare Tunnel con cuenta) sin ese techo, cambiá el comando y el
# patrón de detección de URL acá SIN tocar el resto del server:
#
#   REII_TUNNEL_CMD='ngrok http 8000 --log stdout'
#   REII_TUNNEL_URL_REGEX='https://[a-z0-9-]+\.ngrok-free\.app'
#
# Default: idéntico al comportamiento actual (cloudflared quick tunnel).
TUNNEL_CMD = shlex.split(
    os.environ.get("REII_TUNNEL_CMD", "cloudflared tunnel --url http://localhost:8000")
)
TUNNEL_URL_REGEX = os.environ.get(
    "REII_TUNNEL_URL_REGEX", r"https://[a-z0-9-]+\.trycloudflare\.com"
)

if AUTO_SIZE_REPLICAS:
    _replicas_cfg_msg = f"N_REPLICAS=auto (max={MAX_REPLICAS}, margen={VRAM_SAFETY_MARGIN_GB}GB)"
else:
    _replicas_cfg_msg = f"N_REPLICAS={_N_REPLICAS_ENV} (manual, sin auto-sizing)"
print(
    f"⚙️  Config: {_replicas_cfg_msg} · FP16={USE_FP16} · "
    f"tunnel_cmd={' '.join(TUNNEL_CMD)!r}",
    flush=True,
)

# Ambos endpoints llaman nlp(...) de forma SÍNCRONA. Dentro de un `async def`
# de FastAPI eso bloquea el event loop de uvicorn — nada más (otra petición,
# /health, el keep-alive del túnel) puede avanzar mientras la GPU trabaja.
# `asyncio.to_thread` saca cada llamada bloqueante del loop (torch libera el
# GIL durante el forward pass); un `asyncio.Lock` POR RÉPLICA sigue
# garantizando que nunca haya dos nlp() tocando la MISMA instancia del
# pipeline a la vez (eso sí puede OOM) — pero con N_REPLICAS>1, réplicas
# distintas SÍ pueden procesar en paralelo real. No usar
# `uvicorn.Config(..., workers=N>1)`: cada worker cargaría sus propias
# réplicas del modelo por separado, sin coordinación de memoria entre ellas.

# ── 2. Cargar Stanza en GPU (Main Thread) ──────────────────────────────────────
print("📦 Descargando/Cargando Stanza en GPU...", flush=True)
stanza.download(LANG, processors=PROCESSORS, verbose=False)

_nlps: list = []
_gpu_locks: list = []


def _free_vram_gb() -> tuple[float, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return free_bytes / 1e9, total_bytes / 1e9


def _load_and_probe_replica() -> tuple[object, float, float]:
    """Carga una réplica y le corre una inferencia real para forzar la
    reserva de los buffers de activación/coref — medir solo después de
    ``stanza.Pipeline(...)`` (sin inferencia) subestima la memoria real de
    producción. Devuelve ``(nlp_instance, costo_medido_gb, libres_despues_gb)``.
    """
    torch.cuda.reset_peak_memory_stats()
    free_before, _ = _free_vram_gb()
    nlp_instance = stanza.Pipeline(LANG, processors=PROCESSORS, use_gpu=True, verbose=False)
    nlp_instance(stanza.Document([], text=_VRAM_PROBE_TEXT))  # probe: fuerza el pase completo
    free_after, _ = _free_vram_gb()
    return nlp_instance, free_before - free_after, free_after


if AUTO_SIZE_REPLICAS:
    _, _vram_total = _free_vram_gb()
    print(
        f"🔎 Auto-sizing de réplicas (máx {MAX_REPLICAS}, margen de seguridad "
        f"{VRAM_SAFETY_MARGIN_GB}GB, GPU con {_vram_total:.2f}GB totales)...",
        flush=True,
    )
    for i in range(MAX_REPLICAS):
        free_before, _ = _free_vram_gb()
        print(f"   probando réplica {i + 1}: {free_before:.2f}GB libres antes de cargar...", flush=True)
        try:
            nlp_i, cost_gb, free_after = _load_and_probe_replica()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(
                f"   ❌ CUDA OOM cargando réplica {i + 1} — me quedo con "
                f"{len(_nlps)} réplica(s) ya confirmadas.",
                flush=True,
            )
            torch.cuda.empty_cache()
            break

        if free_after < VRAM_SAFETY_MARGIN_GB and _nlps:
            # Ya hay al menos 1 réplica funcionando: descarto esta por no
            # dejar margen suficiente, y me quedo con las anteriores. No
            # interrumpo el servicio por perseguir una réplica más.
            print(
                f"   ⚠️ réplica {i + 1} costó ~{cost_gb:.2f}GB y dejaría solo "
                f"{free_after:.2f}GB libres (< margen de {VRAM_SAFETY_MARGIN_GB}GB) "
                f"— la descarto, me quedo en {len(_nlps)} réplica(s).",
                flush=True,
            )
            del nlp_i
            gc.collect()
            torch.cuda.empty_cache()
            break

        _nlps.append(nlp_i)
        _gpu_locks.append(asyncio.Lock())
        print(
            f"   ✓ réplica {i + 1} confirmada — costó ~{cost_gb:.2f}GB, "
            f"quedan {free_after:.2f}GB libres",
            flush=True,
        )
        if free_after < VRAM_SAFETY_MARGIN_GB:
            # Caso extremo: ni la primera réplica deja margen de seguridad.
            # La dejamos puesta igual (sin coref el server no sirve para
            # nada), pero no se intenta cargar una segunda.
            print(
                f"   ⚠️ quedan {free_after:.2f}GB libres, por debajo del margen "
                f"({VRAM_SAFETY_MARGIN_GB}GB) ya con 1 sola réplica — no se "
                f"intentan más.",
                flush=True,
            )
            break
    N_REPLICAS = len(_nlps)
    if N_REPLICAS == 0:
        raise RuntimeError(
            "No se pudo cargar ni una réplica de Stanza en la GPU (CUDA OOM "
            "desde la primera). Revisar REII_STANZA_MAX_REPLICAS/memoria "
            "disponible en esta instancia de Colab."
        )
else:
    N_REPLICAS = int(_N_REPLICAS_ENV)
    for i in range(N_REPLICAS):
        _nlps.append(stanza.Pipeline(LANG, processors=PROCESSORS, use_gpu=True, verbose=False))
        _gpu_locks.append(asyncio.Lock())
        if torch.cuda.is_available():
            free_after, total = _free_vram_gb()
            print(
                f"   réplica {i + 1}/{N_REPLICAS} cargada — "
                f"{free_after:.2f}GB libres de {total:.2f}GB totales",
                flush=True,
            )

print(f"✅ Stanza listo en GPU (CUDA) — {N_REPLICAS} réplica(s).", flush=True)

_replica_cycle = itertools.cycle(range(N_REPLICAS))


def _next_replica() -> int:
    # Round-robin simple. Seguro en un solo hilo de event loop: no hay otro
    # código corriendo entre esta llamada y el próximo `await`.
    return next(_replica_cycle)


def subjects_from_doc(doc):
    """Extrae los sujetos sintácticos de las oraciones procesadas."""
    subjects = []
    for sentence in doc.sentences:
        tree = sentence.constituency
        root_ids = [word.id for word in sentence.words if word.head == 0]

        for word in sentence.words:
            if word.head in root_ids and "nsubj" in word.deprel:
                phrase_found = False
                try:
                    leaf = tree.get_leaf_for_index(word.id - 1)
                    curr = leaf
                    while curr.parent is not None:
                        if curr.label == "NP":
                            subjects.append(" ".join(curr.leaf_labels()))
                            phrase_found = True
                            break
                        curr = curr.parent
                except Exception:
                    phrase_found = False

                if not phrase_found:
                    subjects.append(word.text)

    return list(set([s.strip() for s in subjects if s]))


def coref_chains_from_doc(doc, text):
    """Cadenas de correferencia con offsets de caracteres relativos a ``text``.

    Espejo de ``ProgressiveSegmenter._extract_chains_from_doc``: cada mención se
    localiza por (sentence, start_word, end_word) y se traduce a offsets de
    caracteres con ``start_char``/``end_char`` de las palabras límite.
    """
    chains = []
    for chain in doc.coref:
        mentions = []
        for mention in chain.mentions:
            # Índices defensivos (MWT / contracciones en español)
            s_idx = (
                mention.sentence[0]
                if isinstance(mention.sentence, (tuple, list))
                else mention.sentence
            )
            sw_idx = (
                mention.start_word[0]
                if isinstance(mention.start_word, (tuple, list))
                else mention.start_word
            )
            ew_idx = (
                mention.end_word[-1]
                if isinstance(mention.end_word, (tuple, list))
                else mention.end_word
            )

            sent = doc.sentences[s_idx]
            first_word = sent.words[sw_idx]
            last_word = sent.words[ew_idx - 1]

            m_start = first_word.start_char
            if m_start is None and getattr(first_word, "parent", None):
                m_start = first_word.parent.start_char

            m_end = last_word.end_char
            if m_end is None and getattr(last_word, "parent", None):
                m_end = last_word.parent.end_char

            if m_start is None or m_end is None:
                continue

            mentions.append(
                {
                    "text": text[m_start:m_end],
                    "start_char": m_start,
                    "end_char": m_end,
                }
            )
        if mentions:
            chains.append({"mentions": mentions})
    return chains


app = FastAPI(title="REII Stanza Root Extraction")


class RootRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=500)
    lang: str = LANG


class RootResult(BaseModel):
    text: str
    subjects: list[str]


class RootResponse(BaseModel):
    results: list[RootResult]

CHUNK_SIZE = 15  # Procesa de 15 en 15 textos para evitar OOM y timeouts


def _run_roots_chunk(chunk_texts: list[str], nlp_instance) -> list[RootResult]:
    # SÍ es seguro usar nlp(lista, ...) acá porque ROOT_PROCESSORS no incluye
    # coref — ver nota junto a ROOT_PROCESSORS y el comentario largo en
    # /v1/extract_coref sobre por qué ese mismo patrón NO es seguro cuando
    # coref está en la lista de processors.
    in_docs = [stanza.Document([], text=t) for t in chunk_texts]
    try:
        with _maybe_autocast():
            out_docs = nlp_instance(in_docs, processors=ROOT_PROCESSORS)
    except Exception as e:
        # El lote completo falló → procesar uno a uno para aislar el texto malo
        print(f"⚠️ Lote de {len(chunk_texts)} falló ({e}); uno a uno", flush=True)
        out_docs = []
        for t in chunk_texts:
            try:
                with _maybe_autocast():
                    out_docs.append(
                        nlp_instance(stanza.Document([], text=t), processors=ROOT_PROCESSORS)
                    )
            except Exception as e2:
                print(f"⚠️ Texto falló ({e2}); sujetos vacíos", flush=True)
                out_docs.append(stanza.Document([], text=""))

    out = []
    # Un texto malo → sujetos vacíos, no tumba el lote
    for t, doc in zip(chunk_texts, out_docs):
        try:
            subjects = subjects_from_doc(doc)
        except Exception:
            subjects = []
        out.append(RootResult(text=t, subjects=subjects))
    return out


async def _process_roots_chunk(chunk_texts: list[str]) -> list[RootResult]:
    idx = _next_replica()
    async with _gpu_locks[idx]:
        return await asyncio.to_thread(_run_roots_chunk, chunk_texts, _nlps[idx])


@app.post("/v1/extract_roots", response_model=RootResponse)
async def extract_roots(req: RootRequest, authorization: str = Header(default="")):
    # Validar Token
    token = authorization.replace("Bearer ", "").strip()
    if not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Token inválido")

    if req.lang != LANG:
        raise HTTPException(
            status_code=400,
            detail=f"Idioma no soportado: {req.lang!r} (servidor en {LANG!r})",
        )

    # Micro-lotes de CHUNK_SIZE, repartidos round-robin entre réplicas: con
    # N_REPLICAS>1 varios chunks corren en paralelo real (cada uno en su
    # propia instancia/lock); con N_REPLICAS=1 se comporta igual que antes
    # (un chunk a la vez, en la única réplica).
    chunks = [req.texts[i : i + CHUNK_SIZE] for i in range(0, len(req.texts), CHUNK_SIZE)]
    chunk_results = await asyncio.gather(*[_process_roots_chunk(c) for c in chunks])

    results = [r for chunk_out in chunk_results for r in chunk_out]
    return RootResponse(results=results)


class CorefRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=500)
    lang: str = LANG


class CorefMention(BaseModel):
    text: str
    start_char: int
    end_char: int


class CorefChain(BaseModel):
    mentions: list[CorefMention]


class CorefResult(BaseModel):
    text: str
    chains: list[CorefChain]


class CorefResponse(BaseModel):
    results: list[CorefResult]


# NO cambiar esto a nlp(lista_de_docs, processors=...) como hace
# /v1/extract_roots — verificado contra el código real de
# stanza 1.15.0 (pipeline/processor.py:242 UDProcessor.bulk_process +
# pipeline/coref_processor.py CorefProcessor.process):
#   1. bulk_process aplana TODAS las sentences de TODOS los docs de la
#      lista en un único `combined_doc` y corre el processor sobre ESE
#      combined_doc. CorefProcessor.process no conoce fronteras entre
#      documentos: construye sent_ids/cased_words recorriendo
#      `document.sentences` tal cual se lo pasan, así que encadena
#      menciones de textos distintos como si fueran correferentes.
#   2. El resultado (`document.coref = clusters`) queda escrito en el
#      `combined_doc` efímero, NO en los Document originales — así que
#      ni siquiera se recupera por texto: `coref_chains_from_doc(doc, t)`
#      fallaría con cada doc del lote (su `.coref` nunca se setea).
# Confirmado con una reproducción fiel de ambos métodos sobre datos de
# prueba (dos textos con "Juan" sin relación): el lote mezcla "Juan" de
# un texto con "Juan" del otro, y .coref queda vacío en los docs
# originales. Por eso cada texto se procesa en su propia llamada nlp() —
# acá la "paralelización" es entre réplicas (una ventana por réplica a la
# vez), nunca varias ventanas dentro de la misma llamada nlp().
def _run_coref_one(t: str, nlp_instance) -> CorefResult:
    try:
        with _maybe_autocast():
            doc = nlp_instance(stanza.Document([], text=t))
        chains = coref_chains_from_doc(doc, t)
    except Exception as e:
        # Un texto malo → cadenas vacías, no tumba el lote
        print(f"⚠️ Texto falló en coref ({e}); cadenas vacías", flush=True)
        chains = []
    return CorefResult(text=t, chains=chains)


async def _process_coref_one(t: str) -> CorefResult:
    idx = _next_replica()
    async with _gpu_locks[idx]:
        return await asyncio.to_thread(_run_coref_one, t, _nlps[idx])


@app.post("/v1/extract_coref", response_model=CorefResponse)
async def extract_coref(req: CorefRequest, authorization: str = Header(default="")):
    # Validar Token
    token = authorization.replace("Bearer ", "").strip()
    if not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Token inválido")

    if req.lang != LANG:
        raise HTTPException(
            status_code=400,
            detail=f"Idioma no soportado: {req.lang!r} (servidor en {LANG!r})",
        )

    # Cada texto se reparte round-robin entre réplicas y corre en su propio
    # hilo: con N_REPLICAS>1, hasta N_REPLICAS ventanas distintas se resuelven
    # EN PARALELO (cada una su propia llamada nlp(), nunca mezcladas — ver
    # nota arriba). Con N_REPLICAS=1 esto se reduce exactamente al
    # comportamiento anterior: un texto a la vez, serializado por el único
    # lock. El tiempo de GPU por texto individual no cambia (eso solo lo
    # toca FP16, opción 4); lo que gana es paralelismo entre textos.
    results = await asyncio.gather(*[_process_coref_one(t) for t in req.texts])

    return CorefResponse(results=list(results))


@app.get("/health")
def health():
    free_gb, total_gb = _free_vram_gb() if torch.cuda.is_available() else (None, None)
    return {
        "status": "ok",
        "model_loaded": len(_nlps) > 0,
        "n_replicas": N_REPLICAS,
        "n_replicas_auto_sized": AUTO_SIZE_REPLICAS,
        "fp16": USE_FP16,
        "vram_free_gb": free_gb,
        "vram_total_gb": total_gb,
    }


# ── 3. Lanzar Tunnel en segundo plano y Uvicorn en Event Loop de Colab ─────────
print(f"🌐 Iniciando túnel ({' '.join(TUNNEL_CMD)})...", flush=True)
proc = subprocess.Popen(
    TUNNEL_CMD,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

public_url = None
for _ in range(120):
    line = proc.stdout.readline()
    m = re.search(TUNNEL_URL_REGEX, line)
    if m:
        public_url = m.group(0)
        break

print("\n" + "=" * 60)
print("🚀 SERVIDOR Y MODELO LISTOS EN COLAB")
print("PUBLIC URL:", public_url)
print("TOKEN:     ", API_TOKEN)
print("=" * 60 + "\n")

# En lugar de uvicorn.run(), nos acoplamos al event loop ya activo en Colab
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)

# Ejecución bloqueante asíncrona directa en la celda
await server.serve()
