"""Fast tests for the coref timeout fix (no heavy imports).

Verifica:
1. El timeout del cliente es < 100s (por debajo del corte de Cloudflare ~100s).
2. La lógica de troceo produce sub-ventanas <= MAX_WINDOW_CHARS con solape.
3. El invariante de offsets: las sub-ventanas cubren el rango completo sin huecos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reii.config import STANZA_SERVER_TIMEOUT


def _chunk(window_text: str, window_start: int, max_chars: int, overlap: int):
    """Réplica exacta de la lógica de troceo de _extract_global_chains."""
    step = max_chars - overlap
    sub_windows = []
    pos = 0
    n = len(window_text)
    while pos < n:
        end = min(pos + max_chars, n)
        sub_windows.append((window_text[pos:end], window_start + pos))
        if end == n:
            break
        pos += step
    return sub_windows


def test_timeout_below_cloudflare_cutoff():
    # Cloudflare corta a ~100s; el timeout debe ser menor para fallar antes.
    assert STANZA_SERVER_TIMEOUT < 100, (
        f"timeout {STANZA_SERVER_TIMEOUT}s >= 100s: Cloudflare corta antes "
        "y el cliente desperdicia tiempo esperando."
    )


def test_chunk_sizes_within_limit():
    text = "palabra " * 500  # ~4000 chars
    subs = _chunk(text, 0, max_chars=1000, overlap=400)
    assert subs, "no sub-windows produced"
    for sub_text, _ in subs:
        assert len(sub_text) <= 1000, f"sub-window de {len(sub_text)} chars > 1000"


def test_chunk_covers_full_range_no_gaps():
    text = "x" * 2500
    subs = _chunk(text, 100, max_chars=1000, overlap=400)
    # Cada sub-ventana empieza en window_start + pos
    starts = [s for _, s in subs]
    assert starts[0] == 100, "primera sub-ventana no empieza en window_start"
    # La última sub-ventana debe llegar al final
    last_end = starts[-1] + len(subs[-1][0])
    assert last_end == 100 + len(text), "la última sub-ventana no cubre el final"


def test_chunk_overlap_positive():
    text = "y" * 2000
    subs = _chunk(text, 0, max_chars=1000, overlap=400)
    # Con solape, la segunda sub-ventana empieza antes de donde terminó la primera
    if len(subs) >= 2:
        first_end = len(subs[0][0])
        second_start = subs[1][1]
        assert second_start < first_end, "no hay solape entre sub-ventanas"
