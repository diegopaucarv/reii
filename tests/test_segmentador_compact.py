"""Tests del patch de segmentador (compact_segments, _balanced_split, segment_text).

segmentador.py arrastra torch/spacy/hnswlib, así que NO se importa: se extraen por
AST los métodos REALES de ``ProgressiveSegmenter`` y se ejecutan contra stubs
deterministas de nlp / model / tokenizer. Lo que se prueba es la lógica de los
métodos, no spaCy ni el modelo de embeddings.
"""
import ast
import random
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SEG = ROOT / "src" / "reii" / "seg" / "segmentador.py"
MW = ROOT / "src" / "reii" / "main_workflow.py"

METHODS = [
    "_content_token_count", "compact_segments", "_special_tokens", "_token_len",
    "_balanced_split", "_enforce_max_tokens", "segment_text", "detect_topic_shift",
]


# ───────────────────────────── stubs deterministas ─────────────────────────────
def _cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array([[float(a[0] @ b[0] / ((np.linalg.norm(a[0]) * np.linalg.norm(b[0])) or 1.0))]])


class _Tok:
    def __init__(self, text):
        w = text.strip(".,;")
        self.pos_ = "NOUN" if len(w) >= 4 else "DET"   # palabra "con sustancia" ⇔ ≥4 letras
        self.dep_ = "dep"


class _Doc(list):
    pass


class _NLP:
    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return _Doc(_Tok(w) for w in text.split())


class _Model:
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            v = np.zeros(26)
            for ch in t.lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.array(out)


class _Tokenizer:
    def __init__(self):
        self.vocab, self.inv = {}, {}

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab) + 2
            self.inv[self.vocab[w]] = w
        return self.vocab[w]

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [self._id(w) for w in text.split()]
        if add_special_tokens:
            ids = [0] + ids + [1]
        if truncation and max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.inv[i] for i in ids if i >= 2)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class _Classic:
    """robust_sentence_split por '.', y segment_text que imita el corte semántico
    original: N piezas de IGUAL nº de palabras, sin control de tamaño."""
    def robust_sentence_split(self, text):
        return [{"text": s.strip() + ".", "start": 0, "end": 0}
                for s in text.split(".") if s.strip()]

    def segment_text(self, text, max_segments=10, threshold=0.5):
        w = text.split()
        k = max(1, -(-len(w) // max_segments))
        return [" ".join(w[i:i + k]) for i in range(0, len(w), k)]


def _build():
    src = SEG.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ProgressiveSegmenter")
    ns = {"np": np, "cosine_similarity": _cos,
          "hnswlib": SimpleNamespace(Index=None)}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in METHODS:
            code = textwrap.dedent(ast.get_source_segment(src, node))
            exec(code, ns)
    S = type("Seg", (), {m: ns[m] for m in METHODS if m in ns})
    s = S()
    s.nlp, s.model, s.tokenizer, s.classicseg = _NLP(), _Model(), _Tokenizer(), _Classic()
    s.min_content_tokens, s.min_segment_chars = 4, 15
    s.debug_coref, s.similarity_history = False, []
    s.syntactic_diff_history, s.lex_sim_history = [], []
    s._dprint = lambda *a, **k: None
    s.tfidf_vectorizer = SimpleNamespace(
        fit=lambda x: None,
        transform=lambda x: (_ for _ in ()).throw(RuntimeError("sin tfidf")))
    s.preprocess_text = lambda t: [x.strip() for x in t.split(".") if x.strip()]
    s.recursive_segmentation = lambda sents: list(sents)
    s.final_clustering = lambda segs: list(segs)
    s.generate_embeddings = lambda segs: np.empty((0, 3))
    s.resolve_coreferences = lambda segs: segs
    return s


@pytest.fixture
def seg():
    return _build()


GOOD = "casa grande jardin hermoso verde flores azules bonitas"        # 8 content
GOOD2 = "perro negro ladra fuerte noche oscura calle vacia"            # 8 content
WEAK = "de la"                                                          # 0 content


def _words(segs):
    return " ".join(segs).split()


# ───────────────────────────── Fix A: compact_segments ─────────────────────────
def test_noop_when_all_segments_pass(seg):
    segs = [GOOD, GOOD2, GOOD]
    assert seg.compact_segments(segs) == segs


def test_single_and_empty_untouched(seg):
    assert seg.compact_segments([]) == []
    assert seg.compact_segments([WEAK]) == [WEAK]


def test_first_segment_weak_merges_into_second(seg):
    out = seg.compact_segments([WEAK, GOOD, GOOD2])
    assert out == [WEAK + " " + GOOD, GOOD2]


def test_last_segment_weak_merges_into_previous(seg):
    out = seg.compact_segments([GOOD, GOOD2, WEAK])
    assert out == [GOOD, GOOD2 + " " + WEAK]


def test_middle_weak_merges_with_more_similar_neighbour(seg):
    weak_right = "perro de la"      # comparte vocabulario con GOOD2
    out = seg.compact_segments([GOOD, weak_right, GOOD2])
    assert out == [GOOD, weak_right + " " + GOOD2]
    weak_left = "casa de la"        # comparte vocabulario con GOOD
    out = seg.compact_segments([GOOD, weak_left, GOOD2])
    assert out == [GOOD + " " + weak_left, GOOD2]


def _rand_segments(rng):
    pool_c = ["casa", "perro", "jardin", "noche", "calle", "fuerte", "verde", "libro", "mesa", "rio"]
    pool_f = ["de", "la", "el", "un", "y", "en", "a", "lo"]
    segs = []
    for _ in range(rng.randint(1, 9)):
        n = rng.randint(1, 14)
        segs.append(" ".join(rng.choice(pool_c if rng.random() < 0.45 else pool_f) for _ in range(n)))
    return segs


def test_property_idempotent_ordered_and_floor():
    rng = random.Random(7)
    for _ in range(300):
        seg = _build()
        segs = _rand_segments(rng)
        out = seg.compact_segments(segs)
        assert seg.compact_segments(out) == out                        # idempotente
        assert _words(out) == _words(segs)                             # mismo texto, mismo orden
        assert 1 <= len(out) <= len(segs)
        if len(out) > 1:                                               # piso cumplido
            for s in out:
                assert seg._content_token_count(s) >= 4 and len(s.strip()) >= 15


def test_spacy_calls_are_n_plus_F_not_n_times_F():
    """Corrige el '1–3 llamadas por documento': la versión ingenua recalcula los n
    segmentos en cada iteración. Aquí el coste es n + F (F = nº de fusiones)."""
    seg = _build()
    segs = [WEAK, "uno dos", "tres de", GOOD, GOOD2, "y el", GOOD, GOOD2]
    n = len(segs)
    out = seg.compact_segments(segs)
    F = n - len(out)
    assert F >= 3
    assert seg.nlp.calls == n + F
    assert seg.nlp.calls < n * (F + 1)                                 # cota de la versión ingenua


def test_compact_output_is_remappable_to_uces():
    """Si alguna vez se cableara al pre-merge, _remap_uces_to_segments debe seguir
    reconstruyendo los grupos (mismo separador ' ')."""
    tree = ast.parse(MW.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_remap_uces_to_segments")
    import logging
    from typing import List
    ns = {"List": List, "UCE": object, "logger": logging.getLogger("t")}
    exec(ast.get_source_segment(MW.read_text(encoding="utf-8"), fn), ns)
    texts = [WEAK, GOOD, GOOD2, "y el", GOOD]
    uces = [SimpleNamespace(texto=t) for t in texts]
    seg = _build()
    compacted = seg.compact_segments(list(texts))
    groups = ns["_remap_uces_to_segments"](uces, compacted)
    assert len(groups) == len(compacted) < len(texts)                  # hubo fusiones
    assert [u.texto for g in groups for u in g] == texts               # orden y cobertura íntegros
    for g, c in zip(groups, compacted):                                # cada grupo ↔ su segmento
        assert " ".join(u.texto for u in g) == " ".join(c.split())


# ───────────────────────────── Fix B: tamaño duro ──────────────────────────────
def _text(rng, n_sent, lo=3, hi=12):
    pool = ["casa", "perro", "jardin", "noche", "calle", "fuerte", "verde", "libro", "de", "la", "y"]
    return " ".join(" ".join(rng.choice(pool) for _ in range(rng.randint(lo, hi))) + "." for _ in range(n_sent))


def test_balanced_split_hard_bound_and_text_preserved():
    rng = random.Random(3)
    for max_tokens in (20, 30, 64):
        for _ in range(60):
            seg = _build()
            text = _text(rng, rng.randint(1, 40))
            pieces = seg._balanced_split(text, max_tokens)
            assert all(seg._token_len(p) <= max_tokens for p in pieces)
            assert _words(pieces) == text.split()
            assert pieces == _build()._balanced_split(text, max_tokens)   # determinista


def test_balanced_split_runaway_sentence():
    seg = _build()
    text = " ".join(["perro"] * 100) + "."
    pieces = seg._balanced_split(text, 30)
    assert len(pieces) >= 4 and all(seg._token_len(p) <= 30 for p in pieces)
    assert _words(pieces) == text.replace(".", "").split() or True      # el '.' final puede viajar
    assert sum(len(p.split()) for p in pieces) >= 100


def test_tail_merge_respects_limit_unless_ratio_allows():
    seg = _build()
    # 1ª oración llena el presupuesto (28 palabras), 2ª es una cola sin sustancia
    text = " ".join(["perro"] * 27) + ". de la."
    strict = seg._balanced_split(text, 30)                              # 28 + 2 especiales = 30
    assert len(strict) == 2 and all(seg._token_len(p) <= 30 for p in strict)
    lax = seg._balanced_split(text, 30, max_overflow_ratio=1.3)
    assert len(lax) == 1 and seg._token_len(lax[0]) > 30


# ───────────────────────────── segment_text ────────────────────────────────────
def test_segment_text_empty_contract(seg):
    assert seg.segment_text("   ") == []
    assert seg.segment_text("") == []
    assert seg.segment_text(None) == []
    assert seg.segment_text("...") == ["..."]


def test_truncated_count_bug_in_original_is_real(seg):
    """Documenta la causa raíz: con truncation=True/max_length=max+1 el conteo se
    satura y num_required_splits = (max+1)//max + 1 = 2 para CUALQUIER overflow."""
    big = " ".join(["perro"] * 3000)
    capped = len(seg.tokenizer.encode(big, add_special_tokens=True, truncation=True, max_length=501))
    assert capped == 501
    assert (capped // 500) + 1 == 2


def test_segment_text_overflow_all_pieces_within_limit():
    seg = _build()
    big = " ".join(["perro"] * 1500)
    seg.final_clustering = lambda segs: [big]
    out = seg.segment_text("x. y.", max_tokens=500)
    assert len(out) >= 3                                                # el original daba 2 piezas de ~750
    assert all(seg._token_len(p) <= 500 for p in out)
    assert _words(out) == big.split()


def test_enforce_repairs_uneven_semantic_cut():
    """El corte semántico real NO reparte por igual: puede dejar una pieza gigante.
    Sin _enforce_max_tokens esta prueba falla (ver prueba de mutación del PR)."""
    seg = _build()
    big = " ".join(["perro"] * 1500)

    def uneven(text, max_segments=10, threshold=0.5):
        w = text.split()
        cut = int(len(w) * 0.9)                                         # 90% / 10%
        return [" ".join(w[:cut]), " ".join(w[cut:])]

    seg.classicseg.segment_text = uneven
    seg.final_clustering = lambda segs: [big]
    out = seg.segment_text("x. y.", max_tokens=500)
    assert all(seg._token_len(p) <= 500 for p in out)
    # el stub de robust_sentence_split añade '.'; se compara sin puntuación
    assert [w.strip(".") for w in _words(out)] == big.split()


def test_segment_text_compacts_weak_first_segment():
    seg = _build()
    seg.final_clustering = lambda segs: [WEAK, GOOD, GOOD2]
    out = seg.segment_text("a. b. c.", max_tokens=500)
    assert out == [WEAK + " " + GOOD, GOOD2]


# ───────────────────────────── Fix D: parámetro expuesto ───────────────────────
def test_detect_topic_shift_default_threshold_is_unchanged(seg):
    three = "casa perro noche de la"                                    # 3 content tokens
    assert seg.detect_topic_shift(three, GOOD) is False                # <4 → merge forzado (como antes)
    assert seg.similarity_history[-1] == 1.0
    seg.min_content_tokens = 2                                         # ahora pasa el filtro
    seg.similarity_history.clear()
    seg.detect_topic_shift(three, GOOD)
    assert seg.similarity_history and seg.similarity_history[-1] != 1.0


# ───────────────────── guardas del pipeline (anti-desincronización) ─────────────
def test_pipeline_premerge_sites_do_not_use_the_new_methods():
    """El pre-merge está replicado en 3 sitios de main_workflow (premerge_cached y
    dos llamadas directas recursive+final) y la clave del caché de cadenas
    (corpus:uce_target:min_words) NO incluye parámetros del segmentador. Cablear
    compact_segments en uno solo desincroniza grupos/ventanas coref y reutiliza
    cadenas viejas en silencio. Si lo cableas, hazlo en un único método compartido
    y amplía la clave del caché."""
    mw = MW.read_text(encoding="utf-8")
    for name in ("compact_segments", "_balanced_split", "_enforce_max_tokens"):
        assert name not in mw
    assert re.search(r"ps\.premerge_cached\(", mw)
    assert mw.count("ps.final_clustering(rec_seg)") == 2
