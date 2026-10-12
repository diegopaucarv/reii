"""Sujetos raíz con Stanza: equivalencia con el método original, sin coref y por lotes (con pipeline falso)."""

import random

import pytest

from reii.seg.root_cache import (
    ROOT_PROCESSORS,
    RootSubjectCache,
    StanzaRootExtractor,
    subjects_from_doc,
)


# ── doc/árbol falsos con la misma interfaz que usa el código ────────────────
class Word:
    def __init__(self, id, head, deprel, text):
        self.id, self.head, self.deprel, self.text = id, head, deprel, text


class Node:
    def __init__(self, label, children=(), text=None):
        self.label, self.children, self.text, self.parent = label, list(children), text, None
        for c in self.children:
            c.parent = self

    def leaves(self):
        return [self] if not self.children else [l for c in self.children for l in c.leaves()]

    def leaf_labels(self):
        return [l.text for l in self.leaves()]

    def get_leaf_for_index(self, i):
        return self.leaves()[i]  # IndexError si el árbol tiene menos hojas (desajuste MWT)


class Sentence:
    def __init__(self, words, constituency):
        self.words, self.constituency = words, constituency


class Doc:
    def __init__(self, sentences):
        self.sentences = sentences


def _tree(words, rng):
    leaves = [Node("tok", text=w.text) for w in words]
    spans, i = [], 0
    while i < len(leaves):
        j = min(len(leaves), i + rng.randint(1, 3))
        spans.append(Node(rng.choice(["NP", "VP", "PP", "NP"]), leaves[i:j]))
        i = j
    return Node("S", spans)


def random_doc(rng):
    sents = []
    for _ in range(rng.randint(1, 3)):
        n = rng.randint(1, 9)
        roots = rng.sample(range(1, n + 1), k=min(n, rng.randint(1, 2)))
        words = []
        for k in range(1, n + 1):
            head = 0 if k in roots else rng.randint(1, n)
            dep = "root" if head == 0 else rng.choice(["nsubj", "nsubj:pass", "obj", "det", "nsubj"])
            words.append(Word(k, head, dep, f"{rng.choice(['el','Ana','los','médicos','  ','casa'])}{k}"))
        r = rng.random()
        tree = None if r < 0.1 else _tree(words, rng)
        if r > 0.9 and tree is not None:  # menos hojas que palabras: desajuste de índices
            tree = _tree(words[:-1] or words, rng)
        sents.append(Sentence(words, tree))
    return Doc(sents)


def original_find_subjects_for_roots(get_stanza, text):
    """Copia literal de ``ProgressiveSegmenter.find_subjects_for_roots`` (commit 8068ef1)."""
    subjects = []
    try:
        stanza_pipe = get_stanza()
        if not stanza_pipe:
            return []

        doc = stanza_pipe(text)
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

    except Exception:
        return []


class FakePipe:
    """Pipeline falso: el documento depende solo del texto (como un modelo determinista)."""

    def __init__(self, accept_processors=True):
        self.calls, self.accept = [], accept_processors

    def _doc(self, text):
        return random_doc(random.Random(hash(("seed", text)) & 0xFFFF))

    def __call__(self, text, processors=None):
        if processors is not None and not self.accept:
            raise TypeError("__call__() got an unexpected keyword argument 'processors'")
        self.calls.append(("single", text, processors))
        return self._doc(text)

    def bulk_process(self, texts, processors=None):
        if processors is not None and not self.accept:
            raise TypeError("bulk_process() got an unexpected keyword argument 'processors'")
        self.calls.append(("bulk", list(texts), processors))
        return [self._doc(t) for t in texts]


# ── equivalencia con el método original ─────────────────────────────────────
@pytest.mark.parametrize("seed", range(200))
def test_subjects_from_doc_matches_original_on_random_docs(seed):
    doc = random_doc(random.Random(seed))
    expected = original_find_subjects_for_roots(lambda: (lambda t: doc), "x")
    assert sorted(subjects_from_doc(doc)) == sorted(expected)


def test_original_covers_fallbacks_that_the_tests_exercise():
    """Que la muestra aleatoria realmente pase por NP, palabra suelta y árbol ausente."""
    seen_np = seen_word = 0
    for seed in range(200):
        doc = random_doc(random.Random(seed))
        for s in doc.sentences:
            ids = [w.id for w in s.words if w.head == 0]
            for w in s.words:
                if w.head in ids and "nsubj" in w.deprel:
                    try:
                        leaf = s.constituency.get_leaf_for_index(w.id - 1)
                        c = leaf
                        while c.parent is not None and c.label != "NP":
                            c = c.parent
                        seen_np += c.label == "NP"
                        seen_word += c.label != "NP"
                    except Exception:
                        seen_word += 1
    assert seen_np > 20 and seen_word > 20


# ── sin coref ───────────────────────────────────────────────────────────────
def test_single_skips_coref_and_matches_the_full_pipeline_result():
    pipe = FakePipe()
    ex = StanzaRootExtractor(lambda: pipe, skip_coref=True)
    full = StanzaRootExtractor(lambda: pipe, skip_coref=False)
    text = "El médico dijo algo"
    assert sorted(ex.single(text)) == sorted(full.single(text))
    assert pipe.calls[0][2] == ROOT_PROCESSORS and pipe.calls[1][2] is None
    assert "coref" not in ROOT_PROCESSORS and "constituency" in ROOT_PROCESSORS and "depparse" in ROOT_PROCESSORS


def test_old_stanza_without_processors_kwarg_falls_back_once():
    pipe = FakePipe(accept_processors=False)
    ex = StanzaRootExtractor(lambda: pipe, skip_coref=True)
    assert isinstance(ex.single("Ana vino"), list) and ex.skip_coref is False
    ex.single("Luis fue")
    assert all(c[2] is None for c in pipe.calls)  # tras el primer TypeError ya no se reintenta con processors
    assert ex.batch(["Ana a", "Bea b"]) and pipe.calls[-1][0] == "bulk"


def test_single_returns_empty_on_failure_or_missing_pipeline():
    assert StanzaRootExtractor(lambda: None).single("x") == []

    def boom(text, processors=None):
        raise RuntimeError("fallo")

    assert StanzaRootExtractor(lambda: boom).single("x") == []


# ── lotes ───────────────────────────────────────────────────────────────────
def test_batch_matches_single_preserves_order_and_uses_one_call():
    pipe = FakePipe()
    ex = StanzaRootExtractor(lambda: pipe)
    texts = [f"Texto numero {i} con sujeto" for i in range(10)]
    out = ex.batch(texts)
    singles = [ex.single(t) for t in texts]
    assert [sorted(o) for o in out] == [sorted(s) for s in singles]
    bulk = [c for c in pipe.calls if c[0] == "bulk"]
    assert len(bulk) == 1 and bulk[0][1] == texts and bulk[0][2] == ROOT_PROCESSORS


def test_batch_skips_empty_texts_without_shifting_results():
    pipe = FakePipe()
    ex = StanzaRootExtractor(lambda: pipe)
    out = ex.batch(["", "Ana vino", "   ", "Luis fue"])
    assert out[0] == [] and out[2] == []
    assert sorted(out[1]) == sorted(ex.single("Ana vino")) and sorted(out[3]) == sorted(ex.single("Luis fue"))
    assert ex.batch(["", " "]) == [[], []] and sum(c[0] == "bulk" for c in pipe.calls) == 1  # todo vacío: sin llamada


def test_batch_raises_on_length_mismatch_and_cache_recovers_one_by_one():
    class Short(FakePipe):
        def bulk_process(self, texts, processors=None):
            return super().bulk_process(texts[:-1], processors)

    pipe = Short()
    ex = StanzaRootExtractor(lambda: pipe)
    with pytest.raises(ValueError):
        ex.batch(["Ana a", "Bea b"])
    rc = RootSubjectCache(ex.single, ex.batch, batch_size=4)
    rc.prefetch(["Ana a", "Bea b", "Cai c"])
    assert all(t in rc for t in ("Ana a", "Bea b", "Cai c"))  # el lote falló, se resolvió uno a uno


def test_cache_with_stanza_extractor_end_to_end():
    pipe = FakePipe()
    ex = StanzaRootExtractor(lambda: pipe)
    rc = RootSubjectCache(ex.single, ex.batch, batch_size=8)
    ctx = [f"Frase {i} del corpus" for i in range(20)]
    rc.prefetch(ctx)
    assert sum(c[0] == "bulk" for c in pipe.calls) == 3 and not any(c[0] == "single" for c in pipe.calls)
    rc.prefetch(ctx)  # segundo pase: nada que calcular
    assert sum(c[0] == "bulk" for c in pipe.calls) == 3
    assert rc.get(ctx[0]) == set(ex.single(ctx[0]))


# ── script de verificación con modelos reales (aquí, con pipeline falso) ────
def _load_script():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_roots_equivalence.py"
    spec = importlib.util.spec_from_file_location("check_roots_equivalence", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_script_reports_equivalence_and_detects_differences(tmp_path):
    chk = _load_script()
    texts = [f"Texto {i} con el médico Ana{i}" for i in range(23)]
    rep = chk.compare(FakePipe(), texts, batch_size=5)
    assert rep["n"] == 23 and rep["n_mismatch"] == {"lite_single": 0, "lite_batch": 0}
    assert chk.exit_code(rep) == 0 and rep["skip_coref_effective"] is True
    assert "IGUALES" in chk.render(rep)

    class Different(FakePipe):  # un modelo cuyo resultado cambia al saltar procesadores
        def _doc(self, text):
            return random_doc(random.Random(hash(("otro", text)) & 0xFFFF))

        def __call__(self, text, processors=None):
            d = super().__call__(text, processors)
            return d if processors is None else random_doc(random.Random(len(text)))

    bad = chk.compare(Different(), texts, batch_size=5)
    assert chk.exit_code(bad) == 1 and "HAY DIFERENCIAS" in chk.render(bad) and bad["examples"]["lite_single"]

    old = chk.compare(FakePipe(accept_processors=False), texts[:6], batch_size=3)
    assert old["skip_coref_effective"] is False and "AVISO" in chk.render(old)

    f = tmp_path / "t.txt"
    f.write_text("\n".join(texts) + "\n", encoding="utf-8")
    args = chk.argparse.Namespace(txt=str(f), json=None, dsn=None, n=10, seed=1)
    got = chk.load_texts(args)
    assert len(got) == 10 and set(got) <= set(texts)
