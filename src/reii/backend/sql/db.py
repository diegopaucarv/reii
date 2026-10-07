"""
Connection manager para PostgreSQL. Auto-crea el schema en primer uso.
Uso:

    from reii.backend.sql.db import get_conn
    with get_conn() as conn:
        conn.execute("SELECT ...")
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import psycopg
from psycopg.rows import dict_row

from reii.config import DATABASE_URL

_SCHEMA_PATH = Path(__file__).parent / "schema_pg.sql"
_SCHEMA_VERSION = 2


def _apply_schema(conn: psycopg.Connection) -> None:
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()


def _current_version(conn: psycopg.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) AS v FROM _schema_version").fetchone()
    except psycopg.errors.UndefinedTable:
        return 0
    return int(row["v"]) if row and row["v"] is not None else 0


def connect(dsn: Optional[str] = None) -> psycopg.Connection:
    conn = psycopg.connect(dsn or DATABASE_URL, row_factory=dict_row)
    # autocommit: _current_version()'s SELECT must not leave an open/aborted
    # transaction, or subsequent conn.transaction() blocks won't commit.
    conn.autocommit = True
    if _current_version(conn) < _SCHEMA_VERSION:
        _apply_schema(conn)
    return conn


@contextmanager
def get_conn(dsn: Optional[str] = None) -> Iterator[psycopg.Connection]:
    conn = connect(dsn)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: psycopg.Connection) -> Iterator[psycopg.Connection]:
    with conn.transaction():
        yield conn
