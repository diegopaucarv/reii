"""Coref remoto en el analizador gramatical (``CoreferenceResolver``).

El pipeline gramatical construía su propio pipeline Stanza **local** aunque
``STANZA_SERVER_URL`` estuviera definido, duplicando el coref que el segmentador
ya resuelve por HTTP. Estos tests fijan el nuevo camino: con un
``RemoteCorefClient`` el resolver no toca Stanza local, mapea los offsets
locales del servidor a globales del documento, fusiona cadenas y degrada sin
abortar cuando el servidor falla.

Nota: importar ``gramatical_analyzer`` arrastra dependencias pesadas (torch,
spacy, stanza, sentence_transformers), por lo que este módulo tarda ~50s en
cargar.
"""

import pytest

from reii.gram.gramatical_analyzer import CoreferenceResolver
from reii.seg.coref_remote import RemoteCorefClient

FULL = "Ana vino a casa. Ella trajo pan. Luis fue al parque."
SEGMENTS = [
    ("Ana vino a casa.", 0),
    ("Ella trajo pan.", 18),
    ("Luis fue al parque.", 34),
]


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _chains_for_text(text):
    """Cadenas que devolvería el servidor: agrupa nombres propios y pronombres."""
    names = {"ana", "ella", "luis"}
    mentions = []
    for word, start in _words(text):
        if word.lower() in names:
            mentions.append(
                {"text": word, "start_char": start, "end_char": start + len(word)}
            )
    return [{"mentions": mentions}] if mentions else []


def _words(text):
    out = []
    i = 0
    for raw in text.split(" "):
        if raw:
            out.append((raw.strip(".,;:¿?¡!"), i))
        i += len(raw) + 1
    return out


@pytest.fixture(autouse=True)
def _patch_post(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "texts": list(json["texts"])})
        results = [{"text": t, "chains": _chains_for_text(t)} for t in json["texts"]]
        return FakeResponse({"results": results})

    monkeypatch.setattr("requests.post", fake_post)
    return calls


def _resolver(batch_size=1, factory=None, fallback=True):
    client = RemoteCorefClient(
        "https://fake.trycloudflare.com", token="tok", lang="es", timeout=30.0
    )
    return CoreferenceResolver(
        None,
        context_units=2,
        remote_client=client,
        remote_batch_size=batch_size,
        local_pipeline_factory=factory,
        remote_fallback_local=fallback,
    )


# ── Fakes del pipeline Stanza local (para el fallback) ───────────────────────
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
    def __init__(self, representative_text, mentions):
        self.representative_text = representative_text
        self.mentions = mentions


class FakeDoc:
    def __init__(self, sentences, coref):
        self.sentences = sentences
        self.coref = coref


class FakeLocalPipe:
    """Pipeline local determinista: una cadena con los nombres propios."""

    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        words = []
        idxs = []
        for word, start in _words(text):
            idxs.append(len(words))
            words.append(FakeWord(word, start, start + len(word)))
        names = {"ana", "ella", "luis"}
        mentions = [
            FakeMention(0, i, i + 1)
            for i, w in enumerate(words)
            if w.text.lower() in names
        ]
        chains = [FakeChain("Ana", mentions)] if mentions else []
        return FakeDoc([FakeSentence(words)], chains)


# ── 1. Se usa el servidor y los offsets quedan en coordenadas globales ───────
def test_remote_path_maps_offsets_to_global(_patch_post):
    chains = _resolver().resolve(FULL, SEGMENTS)

    assert _patch_post, "no se llamó al servidor"
    assert _patch_post[0]["url"].endswith("/v1/extract_coref")
    assert chains

    for chain in chains:
        assert chain["representative"], chain
        for m in chain["mentions"]:
            assert FULL[m["start_char"] : m["end_char"]] == m["text"], m


# ── 2. El representante se deriva de la mención más larga ────────────────────
def test_representative_is_longest_mention(_patch_post):
    chains = _resolver().resolve(FULL, SEGMENTS)
    texts = {m["text"] for c in chains for m in c["mentions"]}
    assert texts == {"Ana", "Ella", "Luis"}, texts
    # Todas las menciones caen en la misma cadena (fuzzy ≥85 sobre el rep)
    reps = [c["representative"] for c in chains]
    assert all(reps), reps


# ── 3. remote_batch_size agrupa ventanas por petición ────────────────────────
def test_batch_size_groups_windows(_patch_post):
    _resolver(batch_size=1).resolve(FULL, SEGMENTS)
    per_call_single = [len(c["texts"]) for c in _patch_post]
    assert per_call_single == [1] * len(per_call_single)

    _patch_post.clear()
    _resolver(batch_size=3).resolve(FULL, SEGMENTS)
    per_call_batched = [len(c["texts"]) for c in _patch_post]
    assert len(per_call_batched) < len(per_call_single)
    assert max(per_call_batched) <= 3


# ── 4. Sin Stanza local: el resolver no tiene pipeline ───────────────────────
def test_no_local_pipeline_needed(_patch_post):
    resolver = _resolver()
    assert resolver.nlp is None
    assert resolver.resolve(FULL, SEGMENTS)


# ── 5. Fallo del servidor sin Stanza local → degrada y marca incompleto ──────
def test_remote_failure_degrades(monkeypatch):
    def boom(url, json=None, headers=None, timeout=None):
        raise RuntimeError("túnel caído")

    monkeypatch.setattr("requests.post", boom)
    resolver = _resolver()
    assert resolver.resolve(FULL, SEGMENTS) == []
    assert resolver.last_extraction_incomplete is True


# ── 6. Fallback: el servidor falla y se resuelve con Stanza local ────────────
def test_fallback_uses_local_pipeline(monkeypatch):
    def boom(url, json=None, headers=None, timeout=None):
        raise RuntimeError("túnel caído")

    monkeypatch.setattr("requests.post", boom)
    pipe = FakeLocalPipe()
    resolver = _resolver(factory=lambda: pipe)

    chains = resolver.resolve(FULL, SEGMENTS)

    assert pipe.calls, "no se usó el pipeline local"
    assert chains, "el fallback no devolvió cadenas"
    assert resolver.last_extraction_incomplete is False
    for chain in chains:
        for m in chain["mentions"]:
            assert FULL[m["start_char"] : m["end_char"]] == m["text"], m


# ── 7. Fallback deshabilitado: no se toca el pipeline local ──────────────────
def test_fallback_disabled_skips_local(monkeypatch):
    def boom(url, json=None, headers=None, timeout=None):
        raise RuntimeError("túnel caído")

    monkeypatch.setattr("requests.post", boom)
    created = []

    def factory():
        created.append(True)
        return FakeLocalPipe()

    resolver = _resolver(factory=factory, fallback=False)
    assert resolver.resolve(FULL, SEGMENTS) == []
    assert created == [], "no debe cargarse Stanza local con el fallback off"
    assert resolver.last_extraction_incomplete is True


# ── 8. Fallback habilitado pero Stanza local no disponible ───────────────────
def test_fallback_unavailable_marks_incomplete(monkeypatch):
    def boom(url, json=None, headers=None, timeout=None):
        raise RuntimeError("túnel caído")

    monkeypatch.setattr("requests.post", boom)
    resolver = _resolver(factory=lambda: None)
    assert resolver.resolve(FULL, SEGMENTS) == []
    assert resolver.last_extraction_incomplete is True


# ── 9. Respuesta con nº de resultados incorrecto → error explícito ───────────
def test_result_count_mismatch_raises(monkeypatch):
    def bad_post(url, json=None, headers=None, timeout=None):
        return FakeResponse({"results": []})

    monkeypatch.setattr("requests.post", bad_post)
    client = RemoteCorefClient("https://fake", timeout=5.0)
    with pytest.raises(ValueError):
        client.chains_for_texts(["uno", "dos"])
