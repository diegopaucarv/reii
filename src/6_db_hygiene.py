#!/usr/bin/env python3
"""
Script 6 - Higiene de la BD de Optuna (SQLite) de REII.

Standalone: NO importa nada de `reii` (ese paquete solo existe dentro de Docker).
Solo usa sqlite3 de la stdlib y, para el borrado principal, optuna del host.
Si optuna no es importable, cae a un borrado sqlite3 puro respetando el orden
de hijas -> padre.

El nombre del estudio lo sella `_normalize_storage` de main_workflow.py con el
fingerprint del corpus y se crea con load_if_exists=True: reutilizar un nombre
es un warm start contaminado. Por eso hay que borrar los estudios viejos.

El centinela es el tuple de objetivos (-2.0, -1.0, 0.0), guardado en
`trial_values` (NO en `trials.value`, que es NULL por ser multiobjetivo).

Modos:
  --inventory     (por defecto) resumen por estudio: trials, estados,
                  centinelas, validos, user attrs y ruta del .db
  --backup        copia el .db a .bak-<ts> verificando tamano; tambien
                  best_params.json y un zip de data/diag si existen
  --delete P [P]  borra los estudios cuyo study_name contenga algun patron
  --verify        PRAGMA integrity_check + estudios restantes
  --vacuum        VACUUM tras el borrado e informa tamano antes/despues
  --dry-run       muestra que borraria sin borrar

Orden efectivo (independiente del orden de los flags):
  inventory -> backup -> delete -> vacuum -> verify

El borrado exige un backup hecho en la misma invocacion o un .bak-* de menos
de 10 minutos; si no, aborta con codigo 3. Nunca se borra sin backup.

Uso:
  python src/6_db_hygiene.py --inventory
  python src/6_db_hygiene.py --backup --delete reii_search_v3 --vacuum --verify
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
DEFAULT_DB = DATA_DIR / "reii_optuna.db"
BEST_PARAMS = "best_params.json"
DIAG_DIR = "diag"
SENTINEL = (-2.0, -1.0, 0.0)
SENTINEL_TOL = 1e-9
RECENT_BACKUP_SECONDS = 600
EXIT_NO_BACKUP = 3

TRIAL_TABLES = (
    "trial_values",
    "trial_intermediate_values",
    "trial_heartbeats",
    "trial_params",
    "trial_user_attributes",
    "trial_system_attributes",
)
STUDY_TABLES = (
    "study_directions",
    "study_system_attributes",
    "study_user_attributes",
)


def _utf8_stdout() -> None:
    """cp1252 no entiende todos los glifos; si se puede, emitir UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{n} B"
        size /= 1024.0
    return f"{n} B"


def _connect(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db))


def _checkpoint(db: Path) -> None:
    """Pliega un WAL pendiente para que copiar el archivo sea consistente."""
    wal = db.with_name(db.name + "-wal")
    try:
        if wal.exists() and wal.stat().st_size > 0:
            con = _connect(db)
            try:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                con.commit()
            finally:
                con.close()
    except sqlite3.Error as exc:
        print(f"  [!] no se pudo hacer checkpoint del WAL: {exc}")


def _study_values(con: sqlite3.Connection):
    """trial_id -> tuple de objetivos ordenados por indice de objetivo."""
    vals: dict[int, dict[int, float]] = {}
    for tid, obj, value in con.execute(
        "SELECT trial_id, objective, value FROM trial_values"
    ):
        if value is None:
            continue
        vals.setdefault(tid, {})[int(obj)] = float(value)
    return {tid: tuple(v[i] for i in sorted(v)) for tid, v in vals.items()}


def _is_sentinel(obj: tuple) -> bool:
    return len(obj) == len(SENTINEL) and all(
        abs(obj[i] - SENTINEL[i]) <= SENTINEL_TOL for i in range(len(SENTINEL))
    )


def inventory(db: Path) -> None:
    print("=" * 72)
    print("INVENTARIO DE LA BD DE OPTUNA")
    print("=" * 72)
    print(f"BD: {db}")
    if not db.exists():
        print(f"[X] no existe {db}")
        return
    size = db.stat().st_size
    print(f"Tamano: {_human(size)} ({size} bytes)")

    con = _connect(db)
    try:
        studies = con.execute(
            "SELECT study_id, study_name FROM studies ORDER BY study_id"
        ).fetchall()
        if not studies:
            print("[ok] la BD no contiene estudios.")
            return
        print(f"Estudios: {len(studies)}")
        values = _study_values(con)
        for sid, name in studies:
            tids = [
                r[0]
                for r in con.execute(
                    "SELECT trial_id FROM trials WHERE study_id=? ORDER BY trial_id",
                    (sid,),
                )
            ]
            states = dict(
                con.execute(
                    "SELECT state, COUNT(*) FROM trials WHERE study_id=? "
                    "GROUP BY state ORDER BY state",
                    (sid,),
                ).fetchall()
            )
            dirs = [
                str(d[0])
                for d in con.execute(
                    "SELECT direction FROM study_directions WHERE study_id=? "
                    "ORDER BY objective",
                    (sid,),
                )
            ]
            attrs = dict(
                con.execute(
                    "SELECT ua.key, COUNT(*) FROM trial_user_attributes ua "
                    "JOIN trials t ON t.trial_id=ua.trial_id "
                    "WHERE t.study_id=? GROUP BY ua.key ORDER BY ua.key",
                    (sid,),
                ).fetchall()
            )
            n_sent = 0
            n_valued = 0
            n_valid = 0
            for tid in tids:
                obj = values.get(tid)
                if not obj:
                    continue
                n_valued += 1
                if _is_sentinel(obj):
                    n_sent += 1
                else:
                    n_valid += 1
            print("-" * 72)
            print(f"study_id      : {sid}")
            print(f"study_name    : {name}")
            print(f"trials        : {len(tids)}")
            print(f"estados       : {states}")
            print(f"direcciones   : {dirs or 'n/a'}")
            print(
                f"centinelas    : {n_sent} "
                f"(objetivos == {SENTINEL} tol {SENTINEL_TOL:g})"
            )
            print(f"validos       : {n_valid} (con valores y no centinela)")
            if n_valued < len(tids):
                print(f"sin valores   : {len(tids) - n_valued}")
            if n_valued:
                pct = 100.0 * n_sent / n_valued
                print(f"tasa centinela: {pct:.0f}%")
            print(f"user attrs    : {attrs or 'ninguno'}")
            print(f"bd            : {db}")
        print("-" * 72)
        print("[ok] inventario completado.")
    finally:
        con.close()


def _recent_backup(db: Path) -> Path | None:
    """Ultimo .bak-* del .db, no vacio y con mtime < 10 min."""
    cands = sorted(db.parent.glob(db.name + ".bak-*"))
    now = time.time()
    best = None
    for path in cands:
        try:
            st = path.stat()
        except OSError:
            continue
        if st.st_size <= 0:
            continue
        if now - st.st_mtime > RECENT_BACKUP_SECONDS:
            continue
        if best is None or st.st_mtime > best[0]:
            best = (st.st_mtime, path)
    return best[1] if best else None


def do_backup(db: Path) -> Path:
    """Copia .db (y best_params.json, zip de diag) y verifica tamanos."""
    ts = _ts()
    _checkpoint(db)
    original = db.stat().st_size
    if original <= 0:
        print(f"[X] el original {db} esta vacio; no se hace backup.")
        raise SystemExit(1)

    dest = db.with_name(f"{db.name}.bak-{ts}")
    shutil.copy2(db, dest)
    copied = dest.stat().st_size
    if copied <= 0 or copied != original:
        print(
            f"[X] backup NO verificado: original={original} bytes, "
            f"copia={copied} bytes ({dest})"
        )
        raise SystemExit(1)
    print(f"[ok] backup .db -> {dest} ({_human(copied)}, {copied} bytes)")

    best = db.parent / BEST_PARAMS
    if best.exists():
        bdest = best.with_name(f"{best.name}.bak-{ts}")
        shutil.copy2(best, bdest)
        bsize = bdest.stat().st_size
        if bsize <= 0:
            print(f"[X] backup de {best} quedo vacio ({bdest})")
            raise SystemExit(1)
        print(f"[ok] backup json -> {bdest} ({_human(bsize)})")
    else:
        print(f"[!] {best} no existe; se omite.")

    diag = db.parent / DIAG_DIR
    if diag.is_dir() and any(diag.iterdir()):
        zbase = db.parent / f"diag.bak-{ts}"
        try:
            zpath = Path(
                shutil.make_archive(
                    str(zbase), "zip", root_dir=str(db.parent), base_dir=DIAG_DIR
                )
            )
        except OSError as exc:
            print(f"[!] no se pudo zipear {diag}: {exc}")
            return dest
        if not zpath.exists() or zpath.stat().st_size <= 0:
            print(f"[X] zip de diag no verificado ({zpath})")
            raise SystemExit(1)
        print(f"[ok] backup diag -> {zpath} ({_human(zpath.stat().st_size)})")
    else:
        print(f"[!] {diag} no existe o esta vacio; se omite.")

    return dest


def find_matches(db: Path, patterns: list[str]) -> list[tuple[int, str]]:
    con = _connect(db)
    try:
        rows = con.execute(
            "SELECT study_id, study_name FROM studies ORDER BY study_id"
        ).fetchall()
    finally:
        con.close()
    return [(sid, name) for sid, name in rows if any(p in name for p in patterns)]


def _optuna_delete(db: Path, names: list[str]) -> bool:
    """Borra con optuna.delete_study. Devuelve False si optuna no esta."""
    try:
        import optuna
    except Exception as exc:  # ImportError o cualquier rotura de import
        print(f"[!] optuna no importable ({exc}); se usa modo fallback sqlite3.")
        return False

    storage = f"sqlite:///{db.resolve().as_posix()}"
    print(f"-> storage: {storage}")
    print(f"-> optuna {getattr(optuna, '__version__', '?')}")
    for name in names:
        try:
            optuna.delete_study(study_name=name, storage=storage)
            print(f"[ok] borrado estudio: {name}")
        except KeyError:
            print(f"[!] el estudio ya no existe: {name}")
        except Exception as exc:
            print(f"[X] fallo al borrar {name}: {exc}")
            raise SystemExit(1)
    return True


def _existing_tables(con: sqlite3.Connection) -> set[str]:
    return {
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _sqlite_delete(db: Path, rows: list[tuple[int, str]]) -> None:
    """Fallback: borra filas hijas antes que padre, por study_id."""
    con = _connect(db)
    try:
        tables = _existing_tables(con)
        trial_tables = [t for t in TRIAL_TABLES if t in tables]
        study_tables = [t for t in STUDY_TABLES if t in tables]
        for sid, name in rows:
            for table in trial_tables:
                con.execute(
                    f"DELETE FROM {table} WHERE trial_id IN "
                    "(SELECT trial_id FROM trials WHERE study_id=?)",
                    (sid,),
                )
            con.execute("DELETE FROM trials WHERE study_id=?", (sid,))
            for table in study_tables:
                con.execute(f"DELETE FROM {table} WHERE study_id=?", (sid,))
            con.execute("DELETE FROM studies WHERE study_id=?", (sid,))
            print(f"[ok] borrado estudio (sqlite3): {name} (study_id={sid})")
        con.commit()
    except sqlite3.Error as exc:
        con.rollback()
        print(f"[X] fallback sqlite3 fallo: {exc}")
        raise SystemExit(1)
    finally:
        con.close()


def do_delete(db: Path, rows: list[tuple[int, str]], dry_run: bool) -> None:
    names = [name for _sid, name in rows]
    if dry_run:
        for sid, name in rows:
            print(f"-> [dry-run] borraria study_id={sid} {name}")
        print(f"[ok] dry-run: {len(rows)} estudio(s) coinciden; nada borrado.")
        return
    used_optuna = _optuna_delete(db, names)
    if not used_optuna:
        _sqlite_delete(db, rows)
    else:
        print(f"[ok] {len(names)} estudio(s) borrados con optuna.")


def do_vacuum(db: Path) -> None:
    before = db.stat().st_size if db.exists() else 0
    con = _connect(db)
    try:
        con.execute("VACUUM")
        con.commit()
    except sqlite3.Error as exc:
        print(f"[X] VACUUM fallo: {exc}")
        return
    finally:
        con.close()
    after = db.stat().st_size if db.exists() else 0
    print(
        f"[ok] VACUUM: {_human(before)} -> {_human(after)} "
        f"(liberado {_human(max(0, before - after))})"
    )


def verify(db: Path) -> bool:
    print("=" * 72)
    print("VERIFICACION")
    print("=" * 72)
    print(f"BD: {db}")
    if not db.exists():
        print(f"[X] no existe {db}")
        return False
    con = _connect(db)
    try:
        try:
            result = [str(r[0]) for r in con.execute("PRAGMA integrity_check")]
        except sqlite3.DatabaseError as exc:
            print(f"[X] sqlite invalido: {exc}")
            return False
        integrity_ok = result == ["ok"]
        print(f"integrity_check: {'ok' if integrity_ok else 'FALLO'}")
        if not integrity_ok:
            for line in result:
                print(f"  {line}")
        try:
            studies = con.execute(
                "SELECT study_id, study_name FROM studies ORDER BY study_id"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            print(f"[X] no se pudo leer la tabla studies: {exc}")
            return False
        if not studies:
            print("[ok] 0 estudios restantes.")
        else:
            print(f"[!] quedan {len(studies)} estudio(s):")
            for sid, name in studies:
                n = con.execute(
                    "SELECT COUNT(*) FROM trials WHERE study_id=?", (sid,)
                ).fetchone()[0]
                print(f"  study_id={sid} trials={n} {name}")
        return integrity_ok
    finally:
        con.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help=f"ruta del sqlite (por defecto {DEFAULT_DB})",
    )
    ap.add_argument(
        "--inventory",
        action="store_true",
        help="resumen por estudio (modo por defecto)",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="copia .db/.json/zip de diag y verifica tamanos",
    )
    ap.add_argument(
        "--delete",
        nargs="+",
        metavar="PATTERN",
        help="borra estudios cuyo study_name contenga algun patron",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="integrity_check + estudios restantes",
    )
    ap.add_argument(
        "--vacuum",
        action="store_true",
        help="VACUUM tras el borrado (informa tamano antes/despues)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="muestra que borraria sin borrar",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    db = Path(args.db).resolve()

    if not db.exists():
        print(f"[X] no existe la BD {db}")
        return 1

    no_action = not (
        args.inventory or args.backup or args.delete or args.verify or args.vacuum
    )
    did_delete = False

    if no_action or args.inventory:
        inventory(db)

    backup_path = None
    if args.backup:
        backup_path = do_backup(db)

    if args.delete:
        rows = find_matches(db, args.delete)
        patterns = ", ".join(args.delete)
        print("=" * 72)
        print(f"BORRADO (patrones: {patterns})")
        print("=" * 72)
        if not rows:
            print(f"[!] ningun estudio coincide con: {patterns}")
        elif args.dry_run:
            do_delete(db, rows, dry_run=True)
        else:
            used_backup = backup_path or _recent_backup(db)
            if used_backup is None:
                print(
                    "[X] no hay backup valido en esta invocacion ni un .bak-* "
                    f"de menos de {RECENT_BACKUP_SECONDS // 60} min. "
                    "Nada borrado. Usa --backup."
                )
                return EXIT_NO_BACKUP
            print(f"[ok] backup disponible: {used_backup}")
            do_delete(db, rows, dry_run=False)
            did_delete = True

    if args.vacuum:
        if did_delete:
            do_vacuum(db)
        else:
            print("[!] --vacuum no tiene efecto: no se borro nada.")

    rc = 0
    if args.verify:
        if not verify(db):
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
