"""Persistencia append-only de corridas, asignaciones por UCE, niveles y calibración.

Backends:
  * PostgreSQL (producción): vía ``reii.backend.sql.db.connect`` (psycopg 3).
  * SQLite (pruebas / uso offline): DSN ``sqlite:///ruta.db`` o ``:memory:``.

Las tablas son propias (prefijo ``quality_``) y NO tienen claves foráneas hacia
``uces``: ``Database._save()`` reconstruye ``uces`` en cada corrida y no debe
poder borrar el historial de calidad.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd

DDL: List[str] = [
    """CREATE TABLE IF NOT EXISTS quality_run (
        run_id        TEXT PRIMARY KEY,
        label         TEXT,
        kind          TEXT NOT NULL DEFAULT 'pipeline',
        mode          TEXT,
        seed          INTEGER,
        corpus_hash   TEXT,
        code_version  TEXT,
        config_json   TEXT NOT NULL DEFAULT '{}',
        summary_json  TEXT NOT NULL DEFAULT '{}',
        n_uces        INTEGER NOT NULL DEFAULT 0,
        n_uces_total  INTEGER NOT NULL DEFAULT 0,
        created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS quality_tier_profile (
        profile_hash   TEXT PRIMARY KEY,
        name           TEXT NOT NULL,
        rules_json     TEXT NOT NULL,
        preregistered  INTEGER NOT NULL DEFAULT 0,
        created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS quality_uce (
        run_id               TEXT NOT NULL,
        uce_id               TEXT NOT NULL,
        doc_id               TEXT,
        cluster_core         INTEGER,
        s_acuerdo            DOUBLE PRECISION,
        s_margen             DOUBLE PRECISION,
        p_cal                DOUBLE PRECISION,
        n_comparisons        INTEGER,
        s_source             TEXT,
        projected_cluster_id INTEGER,
        n_tokens             INTEGER,
        n_types              INTEGER,
        comps_json           TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (run_id, uce_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_quality_uce_run ON quality_uce(run_id)",
    """CREATE TABLE IF NOT EXISTS quality_tier_result (
        run_id        TEXT NOT NULL,
        tier_id       TEXT NOT NULL,
        profile_hash  TEXT NOT NULL,
        metrics_json  TEXT NOT NULL,
        created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, tier_id, profile_hash)
    )""",
    """CREATE TABLE IF NOT EXISTS quality_gold (
        run_id         TEXT NOT NULL,
        uce_id         TEXT NOT NULL,
        coder          TEXT NOT NULL,
        is_correct     INTEGER NOT NULL,
        human_cluster  INTEGER,
        created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, uce_id, coder)
    )""",
    """CREATE TABLE IF NOT EXISTS quality_calibration (
        run_id       TEXT NOT NULL,
        version      INTEGER NOT NULL,
        method       TEXT NOT NULL,
        params_json  TEXT NOT NULL,
        n_gold       INTEGER NOT NULL,
        created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, version)
    )""",
    """CREATE TABLE IF NOT EXISTS quality_active (
        scope         TEXT PRIMARY KEY,
        run_id        TEXT NOT NULL,
        tier_id       TEXT NOT NULL,
        profile_hash  TEXT,
        updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS quality_audit (
        audit_id     TEXT PRIMARY KEY,
        run_id       TEXT,
        action       TEXT NOT NULL,
        detail_json  TEXT NOT NULL DEFAULT '{}',
        created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
]


def _py(v: Any) -> Any:
    """Convierte tipos numpy / NaN a tipos nativos aptos para el driver."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, (np.bool_,)):
        return int(bool(v))
    if isinstance(v, bool):
        return int(v)
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def dumps(obj: Any) -> str:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        return str(o)

    return json.dumps(obj, ensure_ascii=False, default=_default, sort_keys=True)


class _Cursor:
    def __init__(self, cur):
        self._cur = cur

    def fetchall(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._cur.fetchall()]

    def fetchone(self) -> Optional[Dict[str, Any]]:
        r = self._cur.fetchone()
        return None if r is None else dict(r)


class _SqliteConn:
    """Adaptador mínimo: acepta placeholders %s y devuelve filas dict."""

    is_sqlite = True

    def __init__(self, path: str):
        # autocommit: cada sentencia suelta se confirma sola (como psycopg con autocommit=True)
        self.raw = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.raw.row_factory = sqlite3.Row

    def execute(self, sql: str, params: Iterable[Any] = ()) -> _Cursor:
        return _Cursor(self.raw.execute(sql.replace("%s", "?"), tuple(params)))

    def executemany(self, sql: str, rows: List[tuple]) -> None:
        self.raw.executemany(sql.replace("%s", "?"), rows)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.raw.execute("BEGIN")
        try:
            yield
            self.raw.execute("COMMIT")
        except Exception:
            self.raw.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.raw.close()


class _PgConn:
    is_sqlite = False

    def __init__(self, conn):
        self.raw = conn

    def execute(self, sql: str, params: Iterable[Any] = ()):
        return self.raw.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: List[tuple]) -> None:
        with self.raw.cursor() as cur:
            cur.executemany(sql, rows)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.raw.transaction():
            yield

    def close(self) -> None:
        self.raw.close()


def resolve_dsn(dsn: Optional[str] = None) -> str:
    if dsn:
        return dsn
    env = os.environ.get("REII_QUALITY_DSN")
    if env:
        return env
    from reii.config import DATABASE_URL  # import tardío

    return DATABASE_URL


class QualityStore:
    def __init__(self, conn):
        self.conn = conn

    # ── construcción ────────────────────────────────────────────────────
    @classmethod
    def open(cls, dsn: Optional[str] = None, ensure: bool = True) -> "QualityStore":
        dsn = resolve_dsn(dsn)
        if dsn.startswith("sqlite:///"):
            conn: Any = _SqliteConn(dsn[len("sqlite:///") :] or ":memory:")
        elif dsn == ":memory:":
            conn = _SqliteConn(":memory:")
        else:
            from reii.backend.sql.db import connect  # psycopg

            conn = _PgConn(connect(dsn))
        store = cls(conn)
        store.dsn = dsn  # type: ignore[attr-defined]
        if ensure:
            store.ensure_schema()
        return store

    def ensure_schema(self) -> None:
        with self.conn.transaction():
            for stmt in DDL:
                self.conn.execute(stmt)

    def close(self) -> None:
        self.conn.close()

    # ── corridas ────────────────────────────────────────────────────────
    def save_run(self, run: Dict[str, Any]) -> str:
        """Upsert idempotente por run_id. Devuelve el run_id."""
        run_id = str(run["run_id"])
        self.conn.execute(
            """INSERT INTO quality_run
               (run_id, label, kind, mode, seed, corpus_hash, code_version,
                config_json, summary_json, n_uces, n_uces_total)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (run_id) DO UPDATE SET
                 label=EXCLUDED.label, kind=EXCLUDED.kind, mode=EXCLUDED.mode,
                 seed=EXCLUDED.seed, corpus_hash=EXCLUDED.corpus_hash,
                 code_version=EXCLUDED.code_version, config_json=EXCLUDED.config_json,
                 n_uces=EXCLUDED.n_uces, n_uces_total=EXCLUDED.n_uces_total""",
            (
                run_id,
                run.get("label"),
                run.get("kind", "pipeline"),
                run.get("mode"),
                _py(run.get("seed")),
                run.get("corpus_hash"),
                run.get("code_version"),
                dumps(run.get("config", {})),
                dumps(run.get("summary", {})),
                int(run.get("n_uces", 0)),
                int(run.get("n_uces_total", run.get("n_uces", 0))),
            ),
        )
        return run_id

    def update_summary(self, run_id: str, summary: Dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE quality_run SET summary_json = %s WHERE run_id = %s",
            (dumps(summary), run_id),
        )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        r = self.conn.execute(
            "SELECT * FROM quality_run WHERE run_id = %s", (run_id,)
        ).fetchone()
        if r is None:
            return None
        r["config"] = json.loads(r.pop("config_json") or "{}")
        r["summary"] = json.loads(r.pop("summary_json") or "{}")
        return r

    def list_runs(self, kind: Optional[str] = None) -> pd.DataFrame:
        sql = "SELECT * FROM quality_run"
        params: List[Any] = []
        if kind:
            sql += " WHERE kind = %s"
            params.append(kind)
        sql += " ORDER BY created_at DESC, run_id"
        rows = self.conn.execute(sql, params).fetchall()
        for r in rows:
            r["summary"] = json.loads(r.pop("summary_json") or "{}")
            r.pop("config_json", None)
        return pd.DataFrame(rows)

    # ── asignaciones por UCE ────────────────────────────────────────────
    _UCE_COLS = (
        "run_id uce_id doc_id cluster_core s_acuerdo s_margen p_cal n_comparisons "
        "s_source projected_cluster_id n_tokens n_types comps_json"
    ).split()

    def save_assignments(self, run_id: str, df: pd.DataFrame) -> int:
        """Reemplaza las asignaciones de ESA corrida (otras corridas no se tocan)."""
        rows = []
        for rec in df.to_dict(orient="records"):
            comps = rec.get("comps") or rec.get("comps_json") or {}
            if not isinstance(comps, str):
                comps = dumps(comps)
            rows.append(
                (
                    run_id,
                    str(rec["uce_id"]),
                    None if rec.get("doc_id") is None else str(rec.get("doc_id")),
                    _py(rec.get("cluster_core")),
                    _py(rec.get("s_acuerdo")),
                    _py(rec.get("s_margen")),
                    _py(rec.get("p_cal")),
                    _py(rec.get("n_comparisons")),
                    rec.get("s_source"),
                    _py(rec.get("projected_cluster_id")),
                    _py(rec.get("n_tokens")),
                    _py(rec.get("n_types")),
                    comps,
                )
            )
        cols = ",".join(self._UCE_COLS)
        ph = ",".join(["%s"] * len(self._UCE_COLS))
        with self.conn.transaction():
            self.conn.execute("DELETE FROM quality_uce WHERE run_id = %s", (run_id,))
            if rows:
                self.conn.executemany(
                    f"INSERT INTO quality_uce ({cols}) VALUES ({ph})", rows
                )
        return len(rows)

    def load_assignments(self, run_id: str) -> pd.DataFrame:
        rows = self.conn.execute(
            "SELECT * FROM quality_uce WHERE run_id = %s ORDER BY uce_id", (run_id,)
        ).fetchall()
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=self._UCE_COLS)
        for c in (
            "cluster_core",
            "projected_cluster_id",
            "n_comparisons",
            "n_tokens",
            "n_types",
        ):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in ("s_acuerdo", "s_margen", "p_cal"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["comps"] = df["comps_json"].map(lambda s: json.loads(s or "{}"))
        return df

    def update_p_cal(self, run_id: str, p_by_uce: Dict[str, float]) -> None:
        rows = [(_py(p), run_id, u) for u, p in p_by_uce.items()]
        with self.conn.transaction():
            self.conn.executemany(
                "UPDATE quality_uce SET p_cal = %s WHERE run_id = %s AND uce_id = %s",
                rows,
            )

    # ── perfiles y resultados por nivel (varios resultados por umbral) ──
    def save_profile(
        self, profile_hash: str, name: str, rules: Dict, prereg: bool
    ) -> None:
        self.conn.execute(
            """INSERT INTO quality_tier_profile (profile_hash, name, rules_json, preregistered)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT (profile_hash) DO NOTHING""",
            (profile_hash, name, dumps(rules), int(prereg)),
        )

    def list_profiles(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM quality_tier_profile ORDER BY created_at, profile_hash"
        ).fetchall()
        for r in rows:
            r["rules"] = json.loads(r.pop("rules_json"))
        return rows

    def save_tier_result(
        self, run_id: str, tier_id: str, profile_hash: str, metrics: Dict[str, Any]
    ) -> None:
        self.conn.execute(
            """INSERT INTO quality_tier_result (run_id, tier_id, profile_hash, metrics_json)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT (run_id, tier_id, profile_hash)
               DO UPDATE SET metrics_json = EXCLUDED.metrics_json""",
            (run_id, tier_id, profile_hash, dumps(metrics)),
        )

    def list_tier_results(
        self, run_id: Optional[str] = None, tier_id: Optional[str] = None
    ) -> pd.DataFrame:
        sql, params, cond = "SELECT * FROM quality_tier_result", [], []
        if run_id:
            cond.append("run_id = %s")
            params.append(run_id)
        if tier_id:
            cond.append("tier_id = %s")
            params.append(tier_id)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY run_id, tier_id, created_at"
        rows = self.conn.execute(sql, params).fetchall()
        for r in rows:
            r["metrics"] = json.loads(r.pop("metrics_json"))
        return pd.DataFrame(rows)

    # ── estándar humano y calibración ───────────────────────────────────
    def save_gold(self, run_id: str, rows: List[Dict[str, Any]]) -> int:
        data = [
            (
                run_id,
                str(r["uce_id"]),
                str(r["coder"]),
                int(bool(r["is_correct"])),
                _py(r.get("human_cluster")),
            )
            for r in rows
        ]
        with self.conn.transaction():
            self.conn.executemany(
                """INSERT INTO quality_gold (run_id, uce_id, coder, is_correct, human_cluster)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (run_id, uce_id, coder) DO UPDATE SET
                     is_correct = EXCLUDED.is_correct, human_cluster = EXCLUDED.human_cluster""",
                data,
            )
        return len(data)

    def load_gold(self, run_id: str) -> pd.DataFrame:
        rows = self.conn.execute(
            "SELECT uce_id, coder, is_correct, human_cluster FROM quality_gold WHERE run_id = %s",
            (run_id,),
        ).fetchall()
        return pd.DataFrame(
            rows, columns=["uce_id", "coder", "is_correct", "human_cluster"]
        )

    def save_calibration(self, run_id: str, cal: Dict[str, Any]) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM quality_calibration WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        version = int(row["v"]) + 1
        self.conn.execute(
            """INSERT INTO quality_calibration (run_id, version, method, params_json, n_gold)
               VALUES (%s,%s,%s,%s,%s)""",
            (
                run_id,
                version,
                cal.get("method", "isotonic"),
                dumps(cal),
                int(cal.get("n", 0)),
            ),
        )
        return version

    def load_calibration(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            """SELECT * FROM quality_calibration WHERE run_id = %s
               ORDER BY version DESC LIMIT 1""",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        cal = json.loads(row["params_json"])
        cal["version"] = int(row["version"])
        return cal

    # ── puntero activo y auditoría ──────────────────────────────────────
    def set_active(
        self, scope: str, run_id: str, tier_id: str, profile_hash: Optional[str] = None
    ) -> None:
        self.conn.execute(
            """INSERT INTO quality_active (scope, run_id, tier_id, profile_hash)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT (scope) DO UPDATE SET run_id = EXCLUDED.run_id,
                 tier_id = EXCLUDED.tier_id, profile_hash = EXCLUDED.profile_hash""",
            (scope, run_id, tier_id, profile_hash),
        )

    def get_active(self, scope: str = "default") -> Optional[Dict[str, Any]]:
        return self.conn.execute(
            "SELECT * FROM quality_active WHERE scope = %s", (scope,)
        ).fetchone()

    def log(
        self, run_id: Optional[str], action: str, detail: Optional[Dict] = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO quality_audit (audit_id, run_id, action, detail_json) VALUES (%s,%s,%s,%s)",
            (uuid.uuid4().hex, run_id, action, dumps(detail or {})),
        )

    def audit(self, run_id: Optional[str] = None) -> pd.DataFrame:
        sql, params = "SELECT * FROM quality_audit", []
        if run_id:
            sql += " WHERE run_id = %s"
            params.append(run_id)
        sql += " ORDER BY created_at"
        return pd.DataFrame(self.conn.execute(sql, params).fetchall())
