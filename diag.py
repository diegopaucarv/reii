#!/usr/bin/env python3
"""Punto de entrada del CLI de diagnóstico REII (host, sin instalar el paquete).

    python diag.py                 # lista los comandos
    python diag.py preflight       # equivalen a `python -m reii.diag.cli ...`
    python diag.py optuna-db --help

Existe porque en el host `reii` NO está instalado (solo dentro de la imagen
Docker, vía `pip install -e .`): este shim añade `src/` a `sys.path` y delega
en `reii.diag.cli`. Con el paquete instalado, o dentro del contenedor, usa
`python -m reii.diag.cli` directamente.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reii.diag.cli import main  # noqa: E402  (tras insertar src/ en sys.path)

if __name__ == "__main__":
    raise SystemExit(main())
