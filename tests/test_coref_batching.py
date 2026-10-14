"""Batching de extracción de cadenas de coref (``_extract_global_chains``).

Verifica que el refactor de ``ProgressiveSegmenter._extract_global_chains``
(procesar sub-ventanas candidatas por lotes vía ``bulk_process``) produce
exactamente las mismas cadenas que el procesamiento una-a-una, respeta el
tamaño de lote, degrada a una-a-una ante fallos y no duplica cadenas entre
ventanas solapadas.

Nota: importar ``segmentador`` arrastra dependencias pesadas (torch, spacy,
stanza, sentence_transformers), por lo que este módulo tarda ~50s en cargar.
"""

import re

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
        self.parent = None  # sin MWT: offsets siempre presentes


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
    """Pipeline determinista: el documento depende solo del texto.

    Agrupa en una cadena todas las apariciones de cada nombre propio
    (mayúscula inicial) o pronombre, con offsets reales dentro del texto.
    """

    def __init__(self):
        self.calls = []  # ("bulk", [textos]) o ("single", texto)

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

    def bulk_process(self, texts):
        self.calls.append(("bulk", list(texts)))
        return [self._doc(t) for t in texts]


class FailingBulkPipe(FakeCorefPipe):
    """``bulk_process`` siempre falla; ``__call__`` funciona (fallback)."""

    def bulk_process(self, texts):
        self.calls.append(("bulk", list(texts)))
        raise RuntimeError("OOM simulado")


class ShortBulkPipe(FakeCorefPipe):
    """``bulk_process`` devuelve menos documentos de los pedidos (longitud incorrecta)."""

    def bulk_process(self, texts):
        self.calls.append(("bulk", list(texts)))
        return [self._doc(t) for t in texts[:-1]]


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_ps(pipe):
    """Instancia ``ProgressiveSegmenter`` sin cargar modelos (solo coref)."""
    ps = object.__new__(ProgressiveSegmenter)
    ps._stanza_pipeline = pipe
    ps.debug_coref = False
    return ps


def make_segments(texts):
    """Réplica de la construcción de ``segments_info``/``full_doc_text`` del caller."""
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


def bulk_calls(pipe):
    return [c for c in pipe.calls if c[0] == "bulk"]


# ── 1. Batching produce cadenas idénticas a una-a-una ────────────────────────
def test_batching_matches_one_at_a_time():
    texts = [
        "Ana vino a casa",
        "Luis fue al parque",
        "María y Ana hablaron",
        "Pedro llegó tarde",
        "Ella se fue",
    ]
    segs, full = make_segments(texts)

    pipe_batch = FakeCorefPipe()
    batched = make_ps(pipe_batch)._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=4
    )

    pipe_single = FakeCorefPipe()
    single = make_ps(pipe_single)._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=1
    )

    assert batched == single
    assert batched  # no vacío: hay nombres propios


# ── 2. Se respeta el tamaño de lote ──────────────────────────────────────────
def test_batch_size_respected():
    names = [
        "Ana",
        "Luis",
        "María",
        "Pedro",
        "Juan",
        "Rosa",
        "Carlos",
        "Elena",
        "Miguel",
        "Sofía",
    ]
    texts = [f"{n} dijo algo" for n in names]
    segs, full = make_segments(texts)
    pipe = FakeCorefPipe()
    make_ps(pipe)._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=3
    )

    # 10 ventanas candidatas -> ceil(10/3) = 4 llamadas bulk
    assert len(bulk_calls(pipe)) == 4
    assert sum(len(c[1]) for c in bulk_calls(pipe)) == 10
    assert not any(c[0] == "single" for c in pipe.calls)


def test_batch_size_one_uses_single_calls():
    names = ["Ana", "Luis", "María", "Pedro", "Juan"]
    texts = [f"{n} dijo algo" for n in names]
    segs, full = make_segments(texts)
    pipe = FakeCorefPipe()
    make_ps(pipe)._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=1
    )
    assert not bulk_calls(pipe)
    assert sum(c[0] == "single" for c in pipe.calls) == 5


# ── 2b. Aviso cuando se fuerza batching (>1): Stanza coref bulk_process ─────
# combina sub-ventanas en un único documento (rompe la ventana y pierde cadenas).
def test_warns_when_batch_size_gt_one(caplog):
    import logging

    texts = ["Ana vino", "Luis fue"]
    segs, full = make_segments(texts)
    pipe = FakeCorefPipe()
    with caplog.at_level(logging.WARNING):
        make_ps(pipe)._extract_global_chains(
            segs, full, context_units=2, coref_batch_size=2
        )
    assert any("bulk_process combina" in r.message for r in caplog.records)


def test_no_warning_when_batch_size_one(caplog):
    import logging

    texts = ["Ana vino", "Luis fue"]
    segs, full = make_segments(texts)
    pipe = FakeCorefPipe()
    with caplog.at_level(logging.WARNING):
        make_ps(pipe)._extract_global_chains(
            segs, full, context_units=2, coref_batch_size=1
        )
    assert not any("bulk_process combina" in r.message for r in caplog.records)


# ── 3. Fallback a una-a-una ante fallos de bulk ──────────────────────────────
def test_fallback_when_bulk_raises():
    texts = ["Ana vino", "Luis fue", "María habló", "Pedro llegó"]
    segs, full = make_segments(texts)

    pipe = FailingBulkPipe()
    got = make_ps(pipe)._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=2
    )

    ref = make_ps(FakeCorefPipe())._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=1
    )
    assert got == ref
    # intentó bulk y luego degradó a una-a-una
    assert bulk_calls(pipe) and any(c[0] == "single" for c in pipe.calls)


def test_fallback_when_bulk_returns_wrong_length():
    texts = ["Ana vino", "Luis fue", "María habló"]
    segs, full = make_segments(texts)

    pipe = ShortBulkPipe()
    got = make_ps(pipe)._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=2
    )

    ref = make_ps(FakeCorefPipe())._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=1
    )
    assert got == ref


# ── 4. Filtro de candidatos: ventanas sin nombres/pronombres se saltan ───────
def test_candidate_filter_skips_windows_without_candidates():
    texts = ["casa grande", "perro negro", "gato blanco", "mesa alta"]
    segs, full = make_segments(texts)
    pipe = FakeCorefPipe()
    chains = make_ps(pipe)._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=2
    )
    assert chains == []
    assert pipe.calls == []  # ninguna llamada a Stanza


# ── 5. Dedup: ventanas solapadas no duplican cadenas ─────────────────────────
def test_dedup_across_overlapping_windows():
    # "Ana" aparece en la ventana 0 (solo seg0) y en la ventana 1 (seg0+seg1),
    # con el mismo span global -> debe quedar una sola cadena "ana".
    texts = ["Ana vino", "Luis fue"]
    segs, full = make_segments(texts)
    pipe = FakeCorefPipe()
    chains = make_ps(pipe)._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=1
    )

    def has_text(c, t):
        return any(m["text"] == t for m in c["mentions"])

    ana = [c for c in chains if has_text(c, "Ana")]
    luis = [c for c in chains if has_text(c, "Luis")]
    assert len(ana) == 1 and len(luis) == 1
    assert len(chains) == 2


# ── 6. MWT: índices como tupla/lista se desenvuelven ─────────────────────────
def test_mwt_tuple_indices_are_unwrapped():
    class MwtMention(FakeMention):
        def __init__(self, sentence, start_word, end_word):
            self.sentence = (sentence,)
            self.start_word = (start_word,)
            self.end_word = (end_word,)

    class MwtPipe(FakeCorefPipe):
        def _doc(self, text):
            doc = super()._doc(text)
            # reempaqueta las menciones con índices en tupla (estilo MWT)
            for chain in doc.coref:
                chain.mentions = [
                    MwtMention(m.sentence, m.start_word, m.end_word)
                    for m in chain.mentions
                ]
            return doc

    texts = ["Ana vino", "Luis fue"]
    segs, full = make_segments(texts)
    pipe = MwtPipe()
    chains = make_ps(pipe)._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=2
    )
    ref = make_ps(FakeCorefPipe())._extract_global_chains(
        segs, full, context_units=2, coref_batch_size=1
    )
    assert chains == ref
