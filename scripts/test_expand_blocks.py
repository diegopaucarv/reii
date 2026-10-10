"""
Test: _expand_vectorized con procesamiento por bloques produce EXACTAMENTE
el mismo output que la versión original de matriz completa.

Se compara la lista de (entity_text, entity_head_lemma, verb_lemma,
frame_fingerprint) sobre un dataset sintético pequeño (50 cabezas, 200 nouns).

Uso (desde la raíz del proyecto):
    python scripts/test_expand_blocks.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from reii.gram.gramatical_analyzer import CorefPredicateAnalyzer, PredicateFrame


# ---------------------------------------------------------------------------
# Versión ORIGINAL (matriz completa) — copiada inline para comparación
# ---------------------------------------------------------------------------
def expand_old(
    base_frames, all_nouns, we_analyzer, top_k_similar, similarity_threshold
):
    if not base_frames or not all_nouns:
        return []
    unique_heads = list(
        {f.entity_head_lemma for f in base_frames if f.entity_head_lemma}
    )
    q_vecs = np.array([we_analyzer.vector(h) for h in unique_heads])
    n_vecs = np.array([n["vector"] for n in all_nouns])
    q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-8
    n_norms = np.linalg.norm(n_vecs, axis=1, keepdims=True) + 1e-8
    sim_matrix = (q_vecs / q_norms) @ (n_vecs / n_norms).T
    head_set = set(h.lower() for h in unique_heads)
    expansions = []
    for h_idx, head in enumerate(unique_heads):
        sims = sim_matrix[h_idx]
        k = min(top_k_similar, len(all_nouns))
        top_idx = np.argpartition(sims, -k)[-k:]
        for n_idx in top_idx:
            if sims[n_idx] < similarity_threshold:
                continue
            noun = all_nouns[n_idx]
            if noun["lemma"] in head_set:
                continue
            neg_flag = "NEG" if noun["grammar"].get("negacion") else ""
            obj_lemma = noun["grammar"].get("obj_lemma", "")
            voice = noun["grammar"].get("voz", "Act")
            fingerprint = " ".join(
                x for x in [noun["verb_lemma"], obj_lemma, voice, neg_flag] if x
            )
            expansions.append(
                PredicateFrame(
                    entity_text=noun["text"],
                    entity_head_lemma=noun["lemma"],
                    entity_start_char=noun["char_start"],
                    entity_end_char=noun["char_end"],
                    chain_representative="",
                    verb_lemma=noun["verb_lemma"],
                    verb_text=noun["verb_text"],
                    verb_start_char=0,
                    verb_end_char=0,
                    voice=voice,
                    tense=noun["grammar"].get("tiempo", ""),
                    mood=noun["grammar"].get("modo", ""),
                    negated=bool(noun["grammar"].get("negacion", False)),
                    frame_fingerprint=fingerprint,
                    uce_id=noun["uce_id"],
                    is_expansion=True,
                    original_entity=head,
                )
            )
    return expansions


def frame_key(f):
    return (
        f.entity_text,
        f.entity_head_lemma,
        f.verb_lemma,
        f.frame_fingerprint,
    )


class FakeWeAnalyzer:
    """Devuelve vectores deterministas por texto."""

    def __init__(self, rng):
        self._vecs = {}
        self._rng = rng

    def vector(self, text):
        if text not in self._vecs:
            self._vecs[text] = self._rng.standard_normal(64)
        return self._vecs[text]


def make_synthetic(n_heads=50, n_nouns=200, seed=42):
    rng = np.random.default_rng(seed)
    we = FakeWeAnalyzer(rng)
    heads = [f"entidad_{i}" for i in range(n_heads)]
    # Temas semánticos: cabezas y sustantivos del mismo tema son similares,
    # de modo que el umbral (0.72) y el top_k sí se ejercitan.
    n_themes = 8
    themes = rng.standard_normal((n_themes, 64))
    base_frames = []
    for i, h in enumerate(heads):
        we._vecs[h] = themes[i % n_themes] + 0.05 * rng.standard_normal(64)
        base_frames.append(
            PredicateFrame(
                entity_text=h,
                entity_head_lemma=h,
                entity_start_char=i,
                entity_end_char=i + len(h),
                chain_representative=h,
                verb_lemma=f"verbo_{i % 7}",
                verb_text=f"verbo_{i % 7}",
                verb_start_char=0,
                verb_end_char=0,
                voice="Act",
                tense="",
                mood="",
                negated=False,
                frame_fingerprint=f"fp_{i}",
                uce_id=f"uce_{i % 5}",
                is_expansion=False,
                original_entity="",
            )
        )

    all_nouns = []
    for i in range(n_nouns):
        lemma = f"sustantivo_{i}"
        we._vecs[lemma] = themes[i % n_themes] + 0.05 * rng.standard_normal(64)
        all_nouns.append(
            {
                "text": lemma,
                "lemma": lemma,
                "vector": we.vector(lemma),
                "context": f"contexto {i}",
                "verb_lemma": f"verbo_{i % 7}",
                "verb_text": f"verbo_{i % 7}",
                "grammar": {
                    "negacion": i % 3 == 0,
                    "obj_lemma": f"obj_{i % 5}",
                    "voz": "Act" if i % 2 == 0 else "Pas",
                    "tiempo": "Pres",
                    "modo": "Ind",
                },
                "char_start": 1000 + i,
                "char_end": 1000 + i + len(lemma),
                "uce_id": f"uce_{i % 5}",
            }
        )
    # Algunos sustantivos duplican lemas de cabezas (ejercita el skip por head_set)
    for i in range(5):
        lemma = heads[i]
        all_nouns.append(
            {
                "text": lemma,
                "lemma": lemma,
                "vector": we.vector(lemma),
                "context": f"contexto dup {i}",
                "verb_lemma": f"verbo_{i % 7}",
                "verb_text": f"verbo_{i % 7}",
                "grammar": {
                    "negacion": False,
                    "obj_lemma": f"obj_{i % 5}",
                    "voz": "Act",
                    "tiempo": "Pres",
                    "modo": "Ind",
                },
                "char_start": 2000 + i,
                "char_end": 2000 + i + len(lemma),
                "uce_id": f"uce_{i % 5}",
            }
        )
    return we, base_frames, all_nouns


def main():
    we, base_frames, all_nouns = make_synthetic(n_heads=50, n_nouns=200, seed=42)

    # Instancia real con el código NUEVO (bloques)
    analyzer = CorefPredicateAnalyzer(
        nlp=None,
        we_analyzer=we,
        sentence_embedder=None,
        similarity_threshold=0.72,
        top_k_similar=10,
    )
    new_expansions = analyzer._expand_vectorized(base_frames, all_nouns)

    # Versión ORIGINAL (matriz completa) inline
    old_expansions = expand_old(
        base_frames,
        all_nouns,
        we,
        top_k_similar=10,
        similarity_threshold=0.72,
    )

    new_keys = [frame_key(f) for f in new_expansions]
    old_keys = [frame_key(f) for f in old_expansions]

    assert len(new_keys) == len(old_keys), (
        f"longitud distinta: new={len(new_keys)} old={len(old_keys)}"
    )
    for i, (nk, ok) in enumerate(zip(new_keys, old_keys)):
        assert nk == ok, f"diferencia en índice {i}: new={nk} old={ok}"

    print(f"OK: {len(new_keys)} expansiones idénticas (50 cabezas, 200 nouns)")
    print("  - orden de resultados idéntico")
    print(
        "  - (entity_text, entity_head_lemma, verb_lemma, frame_fingerprint) idénticos"
    )


if __name__ == "__main__":
    main()
