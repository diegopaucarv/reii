"""Permite ``python -m reii.diag <comando>`` (equivale a ``python -m reii.diag.cli``).

La forma corta es la recomendada: no obliga a recordar que el dispatcher vive en
``cli``. En el host sin `pip install -e .`, usa el shim ``python diag.py``.
"""

from __future__ import annotations

from reii.diag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
