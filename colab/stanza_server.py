!pkill -f uvicorn
!pkill -f cloudflare

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
PROCESSORS = "tokenize,mwt,pos,lemma,depparse,constituency"

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

        # Inferencia Stanza por micro-lote
        in_docs = [stanza.Document([], text=t) for t in chunk_texts]
        try:
            out_docs = nlp(in_docs)
        except Exception as e:
            # El lote completo falló → procesar uno a uno para aislar el texto malo
            print(f"⚠️ Lote de {len(chunk_texts)} falló ({e}); uno a uno", flush=True)
            out_docs = []
            for t in chunk_texts:
                try:
                    out_docs.append(nlp(stanza.Document([], text=t)))
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
