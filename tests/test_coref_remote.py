"""Modo remoto de extracción de cadenas coref (``_extract_coref_remote``).

Verifica que cuando ``STANZA_SERVER_URL`` está definido, ``_extract_global_chains``
envía las sub-ventanas al servidor (Colab FastAPI) vía ``/v1/extract_coref``,
mapea los offsets devueltos a globales y deduplica cadenas; y que ante un fallo
remoto cae a Stanza local (o propaga si ``STANZA_SERVER_FALLBACK_LOCAL`` es False).

Nota: importar ``segmentador`` arrastra dependencias pesadas (torch, spacy,
stanza, sentence_transformers), por lo que este módulo tarda ~50s en cargar.
"""

import re

import pytest

from reii.seg import segmentador as seg_mod
from reii.seg.segmentador import ProgressiveSegmenter

# ── Fakes con la interfaz mínima que usa ``_extract_chains_from_doc`` ─────────
PRONOUNS = {
    "él",
    "ella",
    "ellos",
    "ellas",
    "lo",
    "la",
    "le",
    "les",
    "los",
    "las",
    "se",
    "su",
    "sus",
    "me",
    "te",
    "nos",
    "os",
    "mí",
    "mi",
    "ti",
    "sí",
}


class FakeWord:
    def __init__(self, text, start_char, end_char):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.parent = None


class FakeSentence:
    def __init__(self, words):
        self.words = words


class FakeMention:
    def __init__(self, sentence, start_word, end_word):
        self.sentence = sentence
        self.start_word = start_word
        self.end_word = end_word


class FakeChain:
    def __init__(self, mentions):
        self.mentions = mentions


class FakeDoc:
    def __init__(self, sentences, coref):
        self.sentences = sentences
        self.coref = coref


class FakeCorefPipe:
    """Pipeline determinista: agrupa nombres propios/pronombres en cadenas."""

    def __init__(self):
        self.calls = []

    def _doc(self, text):
        words = []
        mentions_by_key = {}
        for m in re.finditer(r"\S+", text):
            w = m.group(0)
            idx = len(words)
            words.append(FakeWord(w, m.start(), m.end()))
            if w[0].isupper() or w.lower() in PRONOUNS:
                mentions_by_key.setdefault(w.lower(), []).append(idx)
        sentence = FakeSentence(words)
        chains = [
            FakeChain([FakeMention(0, i, i + 1) for i in idxs])
            for idxs in mentions_by_key.values()
        ]
        return FakeDoc([sentence], chains)

    def __call__(self, text):
        self.calls.append(("single", text))
        return self._doc(text)


def make_ps(pipe):
    """Instancia ``ProgressiveSegmenter`` sin cargar modelos (solo coref)."""
    ps = object.__new__(ProgressiveSegmenter)
    ps._stanza_pipeline = pipe
    ps.debug_coref = False
    ps.stanza_lang = "es"
    return ps


def make_segments(texts):
    full_doc_text = ""
    segments_info = []
    current_offset = 0
    for t in texts:
        start = current_offset
        end = start + len(t)
        segments_info.append(
            {"text": t, "start": start, "end": end, "uce_id": f"u{len(segments_info)}"}
        )
        full_doc_text += t + " "
        current_offset = end + 1
    return segments_info, full_doc_text


def _remote_chains_for(text):
    """Réplica del JSON que devolvería el servidor para ``text``."""
    pipe = FakeCorefPipe()
    doc = pipe._doc(text)
    chains = []
    for chain in doc.coref:
        mentions = []
        for m in chain.mentions:
            sent = doc.sentences[m.sentence]
            first = sent.words[m.start_word]
            last = sent.words[m.end_word - 1]
            mentions.append(
                {
                    "text": text[first.start_char : last.end_char],
                    "start_char": first.start_char,
                    "end_char": last.end_char,
                }
            )
        if mentions:
            chains.append({"mentions": mentions})
    return chains


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _remote_mode(monkeypatch):
    """Fuerza el modo remoto para estos tests."""
    monkeypatch.setattr(seg_mod, "STANZA_SERVER_URL", "https://fake.trycloudflare.com")
    monkeypatch.setattr(seg_mod, "STANZA_SERVER_TOKEN", "tok")
    monkeypatch.setattr(seg_mod, "STANZA_SERVER_TIMEOUT", 30.0)
    monkeypatch.setattr(seg_mod, "STANZA_SERVER_FALLBACK_LOCAL", True)


# ── 1. Modo remoto: se usa /v1/extract_coref y se mapean offsets ─────────────
def test_remote_path_used_and_offsets_mapped(monkeypatch):
    texts = ["Ana vino a casa", "Luis fue al parque"]
    segs, full = make_segments(texts)
    pipe = FakeCorefPipe()
    ps = make_ps(pipe)

    posted = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        posted["url"] = url
        posted["texts"] = json["texts"]
        results = [{"text": t, "chains": _remote_chains_for(t)} for t in json["texts"]]
        return FakeResponse({"results": results})

    monkeypatch.setattr(seg_mod.requests, "post", fake_post)

    chains = ps._extract_global_chains(segs, full, context_units=2)

    assert posted["url"] == "https://fake.trycloudflare.com/v1/extract_coref"
    assert posted["texts"]  # se enviaron sub-ventanas
    assert chains  # cadenas mapeadas
    # El pipeline local NO se usó en modo remoto
    assert pipe.calls == []


# ── 2. Fallback a local cuando el servidor falla ──────────────────────────────
def test_remote_failure_falls_back_to_local(monkeypatch):
    texts = ["Ana vino", "Luis fue"]
    segs, full = make_segments(texts)
    pipe = FakeCorefPipe()
    ps = make_ps(pipe)

    def fake_post(url, json=None, headers=None, timeout=None):
        raise RuntimeError("túnel caído")

    monkeypatch.setattr(seg_mod.requests, "post", fake_post)

    chains = ps._extract_global_chains(segs, full, context_units=2)

    # Cae a local: el pipeline local se usó
    assert pipe.calls
    assert chains


# ── 3. Sin fallback local: el fallo remoto propaga (sin cadenas) ──────────────
def test_remote_failure_without_local_fallback(monkeypatch):
    monkeypatch.setattr(seg_mod, "STANZA_SERVER_FALLBACK_LOCAL", False)
    texts = ["Ana vino", "Luis fue"]
    segs, full = make_segments(texts)
    pipe = FakeCorefPipe()
    ps = make_ps(pipe)

    def fake_post(url, json=None, headers=None, timeout=None):
        raise RuntimeError("túnel caído")

    monkeypatch.setattr(seg_mod.requests, "post", fake_post)

    chains = ps._extract_global_chains(segs, full, context_units=2)

    assert chains == []
    assert pipe.calls == []  # nunca se tocó el pipeline local


# ── 4. Respuesta malformada: se ignoran menciones sin offsets ────────────────
def test_remote_malformed_mentions_skipped(monkeypatch):
    texts = ["Ana vino", "Luis fue"]
    segs, full = make_segments(texts)
    pipe = FakeCorefPipe()
    ps = make_ps(pipe)

    def fake_post(url, json=None, headers=None, timeout=None):
        results = [
            {
                "text": t,
                "chains": [
                    {
                        "mentions": [
                            {"text": "Ana", "start_char": None, "end_char": 3},
                            {"text": "Ana", "start_char": 0, "end_char": 3},
                        ]
                    }
                ],
            }
            for t in json["texts"]
        ]
        return FakeResponse({"results": results})

    monkeypatch.setattr(seg_mod.requests, "post", fake_post)

    chains = ps._extract_global_chains(segs, full, context_units=2)

    # Solo la mención con offsets válidos sobrevive
    for c in chains:
        for m in c["mentions"]:
            assert m["start_char"] is not None and m["end_char"] is not None
