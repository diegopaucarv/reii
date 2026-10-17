"""Pre-merge caching: bit-exact comparison of cached vs uncached outputs."""

import pytest

from reii.seg.segmentador import ProgressiveSegmenter


def test_premerge_cache_bit_exact(sample_uces_por_doc):
    """Pre-merge cache must reproduce the same clustering as the uncached path."""
    ps = ProgressiveSegmenter(
        model_name="ibm-granite/granite-embedding-107m-multilingual",
        similarity_threshold=0.6,
        max_depth=3,
        debug_coref=False,
    )
    for doc_uces in sample_uces_por_doc[:3]:
        if len(doc_uces) <= 1:
            continue

        # 1. Path sin caché
        ps._premerge_cache.clear()
        rec = ps.recursive_segmentation([u.texto for u in doc_uces])
        prog_ref = ps.final_clustering(rec)

        # 2. Path con caché, primera invocación (miss)
        ps._premerge_cache.clear()
        prog_cached_1 = ps.premerge_cached(doc_uces)

        # 3. Path con caché, segunda invocación (hit)
        prog_cached_2 = ps.premerge_cached(doc_uces)

        assert prog_ref == prog_cached_1, (
            f"cache MISS no reproduce el output sin caché "
            f"({len(prog_ref)} vs {len(prog_cached_1)} segmentos)"
        )
        assert prog_cached_2 == prog_cached_1, "cache HIT divergió del MISS"

        # 4. Guard: mutar el resultado no envenena el caché
        prog_cached_2.append("SENTINEL")
        prog_cached_3 = ps.premerge_cached(doc_uces)
        assert "SENTINEL" not in prog_cached_3, (
            "el caché devolvió una lista contaminada"
        )
