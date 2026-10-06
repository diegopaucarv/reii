"""
Connection manager para SQLite. Auto-crea el schema en primer uso.
Uso:

    from reii.backend.db import get_conn
    with get_conn() as conn:
        conn.execute("SELECT ...")
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from reii.config import WORKFLOW_SQLITE_PATH  # ver §Fase 0.4

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_SCHEMA_VERSION = 1


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def _current_version(conn: sqlite3.Connection) -> int:
    """
    Devuelve la versión de schema aplicada. En una DB recién creada
    (o en un archivo que aún no es SQLite), la tabla `_schema_version`
    no existe todavía — ese caso es indistinguible de "versión 0" y
    debe devolverse 0 sin lanzar excepción.
    """
    try:
        row = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
    except sqlite3.OperationalError as e:
        # "no such table: _schema_version" → DB nueva, aplicar schema
        if "no such table" in str(e):
            return 0
        # Cualquier otro error operacional sí debe propagarse
        raise
    return int(row[0]) if row and row[0] is not None else 0


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    """
    Abre (y crea si hace falta) la DB. Aplica el schema si no existe
    o si la versión es menor que la actual.
    """
    db_path = path or WORKFLOW_SQLITE_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        isolation_level=None,  # autocommit; gestionamos transacciones explícitas
    )
    conn.row_factory = sqlite3.Row

    # Pragmas por conexión
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    if _current_version(conn) < _SCHEMA_VERSION:
        _apply_schema(conn)

    return conn


@contextmanager
def get_conn(path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """
    Uso:
        with get_conn() as conn:
            rows = conn.execute(...).fetchall()
    Cierra la conexión al salir. Usa BEGIN IMMEDIATE para escritura.
    """
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """
    Transacción explícita con rollback automático en excepción.
    BEGIN IMMEDIATE evita contention con lectores en WAL.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
