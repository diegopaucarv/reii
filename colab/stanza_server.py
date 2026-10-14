!pkill -f uvicorn
!pkill -f cloudflare

# ── 0. Fix de entorno: torchao desactualizado rompe la importación de torch ──
# Colab trae torchao 0.10.0 preinstalado, pero el torch reciente (2.9+) exige
# torchao >= 0.16.0. Sin este upgrade, `import stanza` (que importa torch)
# lanza: ImportError: Found an incompatible version of torchao.
!pip install -U "torchao>=0.16.0" -q

import os
import re
import secrets
import subprocess

import stanza
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# ── 1. Configuración ────────────────────────────────────────────────────────────
API_TOKEN = "6b92bb50d646a9a4030ac1adf0a5cc8e"
LANG = "es"
PROCESSORS = "tokenize,mwt,pos,lemma,depparse,constituency,coref"
# Subconjunto para sujetos raíz: sin coref (mismo patrón que root_cache.py).
ROOT_PROCESSORS = "tokenize,mwt,pos,lemma,depparse,constituency"

# ── 2. Cargar Stanza en GPU (Main Thread) ──────────────────────────────────────
print("📦 Descargando/Cargando Stanza en GPU...", flush=True)
stanza.download(LANG, processors=PROCESSORS, verbose=False)
nlp = stanza.Pipeline(LANG, processors=PROCESSORS, use_gpu=True, verbose=False)
print("✅ Stanza listo en GPU (CUDA).", flush=True)


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

    results = []

    # Subdividir req.texts en micro-lotes para no saturar la GPU ni superar los 100s de Cloudflare
    for i in range(0, len(req.texts), CHUNK_SIZE):
        chunk_texts = req.texts[i : i + CHUNK_SIZE]

        # Inferencia Stanza por micro-lote (sin coref: solo raíces)
        in_docs = [stanza.Document([], text=t) for t in chunk_texts]
        try:
            out_docs = nlp(in_docs, processors=ROOT_PROCESSORS)
        except Exception as e:
            # El lote completo falló → procesar uno a uno para aislar el texto malo
            print(f"⚠️ Lote de {len(chunk_texts)} falló ({e}); uno a uno", flush=True)
            out_docs = []
            for t in chunk_texts:
                try:
                    out_docs.append(
                        nlp(stanza.Document([], text=t), processors=ROOT_PROCESSORS)
                    )
                except Exception as e2:
                    print(f"⚠️ Texto falló ({e2}); sujetos vacíos", flush=True)
                    out_docs.append(stanza.Document([], text=""))

        # Acumular resultados: un texto malo → sujetos vacíos, no tumba el lote
        for t, doc in zip(chunk_texts, out_docs):
            try:
                subjects = subjects_from_doc(doc)
            except Exception:
                subjects = []
            results.append(RootResult(text=t, subjects=subjects))

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

    results = []
    # Cada texto se procesa INDIVIDUALMENTE: bulk_process combinaría todas las
    # sub-ventanas en un único documento y rompería la ventana limitada.
    for t in req.texts:
        try:
            doc = nlp(stanza.Document([], text=t))
            chains = coref_chains_from_doc(doc, t)
        except Exception as e:
            # Un texto malo → cadenas vacías, no tumba el lote
            print(f"⚠️ Texto falló en coref ({e}); cadenas vacías", flush=True)
            chains = []
        results.append(CorefResult(text=t, chains=chains))

    return CorefResponse(results=results)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": nlp is not None}


# ── 3. Lanzar Tunnel en segundo plano y Uvicorn en Event Loop de Colab ─────────
print("🌐 Iniciando Cloudflare Tunnel...", flush=True)
proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

public_url = None
for _ in range(120):
    line = proc.stdout.readline()
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
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
