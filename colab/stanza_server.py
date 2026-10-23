!pkill -f uvicorn
!pkill -f cloudflare

# ── 0. Fix de entorno: torchao desactualizado rompe la importación de torch ──
# Colab trae torchao 0.10.0 preinstalado, pero el torch reciente (2.9+) exige
# torchao >= 0.16.0. Sin este upgrade, `import stanza` (que importa torch)
# lanza: ImportError: Found an incompatible version of torchao.
!pip install -U "torchao>=0.16.0" -q
!pkill -f uvicorn
!pkill -f cloudflare

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

os.environ["HF_TOKEN"] = "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"
os.environ["REII_STANZA_FP16"] = "0"


API_TOKEN = "6b92bb50d646a9a4030ac1adf0a5cc8e"
LANG = "es"
PROCESSORS = "tokenize,mwt,pos,lemma,depparse,constituency,coref"
ROOT_PROCESSORS = "tokenize,mwt,pos,lemma,depparse,constituency"

TUNNEL_CMD = shlex.split(
    os.environ.get("REII_TUNNEL_CMD", "cloudflared tunnel --url http://localhost:8000")
)
TUNNEL_URL_REGEX = os.environ.get(
    "REII_TUNNEL_URL_REGEX", r"https://[a-z0-9-]+\.trycloudflare\.com"
)

print("📦 Descargando/Cargando Stanza en CPU...", flush=True)
stanza.download(LANG, processors=PROCESSORS, verbose=False)

nlp_main = stanza.Pipeline(LANG, processors=PROCESSORS, use_gpu=False, verbose=False)
cpu_lock = asyncio.Lock()


def subjects_from_doc(doc):
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
    chains = []
    for chain in doc.coref:
        mentions = []
        for mention in chain.mentions:
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


CHUNK_SIZE = 15


def _run_roots_chunk(chunk_texts: list[str], nlp_instance) -> list[RootResult]:
    in_docs = [stanza.Document([], text=t) for t in chunk_texts]
    try:
        out_docs = nlp_instance(in_docs, processors=ROOT_PROCESSORS)
    except Exception as e:
        print(f"⚠️ Lote de {len(chunk_texts)} falló ({e}); uno a uno", flush=True)
        out_docs = []
        for t in chunk_texts:
            try:
                out_docs.append(
                    nlp_instance(stanza.Document([], text=t), processors=ROOT_PROCESSORS)
                )
            except Exception as e2:
                print(f"⚠️ Texto falló ({e2}); sujetos vacíos", flush=True)
                out_docs.append(stanza.Document([], text=""))

    out = []
    for t, doc in zip(chunk_texts, out_docs):
        try:
            subjects = subjects_from_doc(doc)
        except Exception:
            subjects = []
        out.append(RootResult(text=t, subjects=subjects))
    return out


async def _process_roots_chunk(chunk_texts: list[str]) -> list[RootResult]:
    async with cpu_lock:
        return await asyncio.to_thread(_run_roots_chunk, chunk_texts, nlp_main)


@app.post("/v1/extract_roots", response_model=RootResponse)
async def extract_roots(req: RootRequest, authorization: str = Header(default="")):
    token = authorization.replace("Bearer ", "").strip()
    if not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Token inválido")

    if req.lang != LANG:
        raise HTTPException(
            status_code=400,
            detail=f"Idioma no soportado: {req.lang!r} (servidor en {LANG!r})",
        )

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


def _run_coref_one(t: str, nlp_instance) -> CorefResult:
    try:
        doc = nlp_instance(stanza.Document([], text=t))
        chains = coref_chains_from_doc(doc, t)
    except Exception as e:
        print(f"⚠️ Texto falló en coref ({e}); cadenas vacías", flush=True)
        chains = []
    return CorefResult(text=t, chains=chains)


async def _process_coref_one(t: str) -> CorefResult:
    async with cpu_lock:
        return await asyncio.to_thread(_run_coref_one, t, nlp_main)


@app.post("/v1/extract_coref", response_model=CorefResponse)
async def extract_coref(req: CorefRequest, authorization: str = Header(default="")):
    token = authorization.replace("Bearer ", "").strip()
    if not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Token inválido")

    if req.lang != LANG:
        raise HTTPException(
            status_code=400,
            detail=f"Idioma no soportado: {req.lang!r} (servidor en {LANG!r})",
        )

    results = await asyncio.gather(*[_process_coref_one(t) for t in req.texts])

    return CorefResponse(results=list(results))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": nlp_main is not None,
        "device": "cpu",
    }


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
print("🚀 SERVIDOR Y MODELO LISTOS (CPU)")
print("PUBLIC URL:", public_url)
print("TOKEN:      ", API_TOKEN)
print("=" * 60 + "\n")

config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)

await server.serve()
