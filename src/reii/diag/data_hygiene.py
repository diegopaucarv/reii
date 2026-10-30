#!/usr/bin/env python3
"""
Higiene de los temporales de ``data/``. Hermano de ``db-hygiene``, que solo
gestiona la BD de Optuna; este se ocupa de los ficheros.

``data/`` esta entero en .gitignore y acumula logs, sondas y checkpoints sin
limite. La politica es DECLARATIVA (``POLICY``): una lista de reglas
``(glob, keep_latest, nota)``. Por cada glob se ordenan sus ficheros por mtime
descendente y se conservan los ``keep_latest`` mas nuevos; el resto pasa a
candidato a borrado. Gana la PRIMERA regla que casa (las mas especificas van
antes), asi que un fichero pertenece a un solo grupo.

Los CANONICOS (``CANONICAL``) no se tocan nunca, aunque un glob los case: el
inventario los marca ``[keep]`` y el borrado los salta. Sus directorios
(``entrevistas/``, ``txt_outputs/``, ``checkpoints/``) tampoco se recorren.

Modos:
  (por defecto)   DRY-RUN: imprime el inventario y que se borraria. Nada muta.
  --apply         borra los candidatos (o los mueve si hay --backup-dir)
  --backup-dir X  con --apply, ``shutil.move`` a X/<ruta relativa> en vez de
                  borrar; se preserva el arbol relativo a data/
  --min-age-minutes MIN
                  un fichero modificado hace menos de MIN minutos se marca
                  ``[skip: reciente]`` y no se toca (default 10)
  --json          inventario como JSON por stdout (los avisos van a stderr)
  --data-dir DIR  override de data/ (util para tests y para otra raiz)

Garantias:
  * Dry-run por defecto: solo ``--apply`` borra o mueve.
  * Canonicos intactos: se comprueban antes y despues de casar el glob.
  * Nunca se sale de DATA_DIR: cada ruta se resuelve y se valida con
    ``is_relative_to``; si no cuelga de la raiz, se rechaza.
  * Los ficheros recientes se respetan, para no pisar una corrida en curso.

Codigos de salida: 0 ok; 1 error de IO; 2 argparse (flag invalido).

Uso:
  python -m reii.diag.cli data-hygiene
  python -m reii.diag.cli data-hygiene --apply --backup-dir ../data_backup
  python -m reii.diag.cli data-hygiene --json | python -m json.tool
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from reii.diag.common import DATA_DIR, utf8_stdout


class Rule(NamedTuple):
    """Una regla de la politica: glob relativo a data/ + cuantos conservar."""

    pattern: str
    keep_latest: int
    nota: str


# Orden = precedencia: gana la primera regla que casa. Lo especifico primero.
POLICY: List[Rule] = [
    Rule("diag/*.dryrun.log", 8, "logs de dry-run"),
    Rule("diag/*.log", 8, "logs de corridas (seed*, bench, select_*)"),
    Rule("diag/tmp_*", 0, "restos de sesion (siempre sobran)"),
    Rule("diag/best_params_seed*.json", 5, "best_params por semilla de trials"),
    Rule("diag/*.json", 12, "sondas y artefactos sueltos (probe*, paper_terms...)"),
    Rule("diag/*.png", 10, "figuras sueltas"),
    Rule("diag/*.html", 10, "informes HTML sueltos"),
    Rule("*.pre_v4", 2, "snapshots previos al tag v4"),
    Rule("*.pre_probe_b", 2, "snapshots previos a la sonda B"),
    Rule("*.pre_script2", 2, "snapshots previos al script 2"),
    Rule("*.pre_script*", 2, "snapshots previos a cualquier script N"),
    Rule("*.ctx*.json", 2, "checkpoints de coref por ventana (ctxN)"),
    Rule("*.round*.json", 2, "rondas de method_selection"),
    Rule("best_params.v3_censurado.json", 1, "copia censurada (una sola)"),
    Rule("comparative_results.json", 1, "comparativa (una sola)"),
]

# Lo que nunca se toca. Un nombre con "/" final es un directorio.
CANONICAL: Tuple[str, ...] = (
    "workflow_data.json",
    "workflow_data.db",
    "best_params.json",
    "method_selection.json",
    "method_selection_log.jsonl",
    "reii_optuna.db",
    "uce_calibration.json",
    "token_table.jsonl",
    "entrevistas.json",
    "SUBTLEX-ESP.xlsx",
    "query_db.py",
    "entrevistas/",
    "txt_outputs/",
    "checkpoints/",
)
CANONICAL_FILES: Tuple[str, ...] = tuple(n for n in CANONICAL if not n.endswith("/"))
CANONICAL_DIRS: Tuple[str, ...] = tuple(n[:-1] for n in CANONICAL if n.endswith("/"))


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{n} B"
        size /= 1024.0
    return f"{n} B"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


def _canonical_name(rel: str) -> Optional[str]:
    """Nombre canonico si ``rel`` (posix, relativo a data/) lo es; si no, None."""
    return rel if rel in CANONICAL_FILES else None


def _matches(rel: str, pattern: str) -> bool:
    """Casa un glob de la politica contra una ruta relativa posix.

    Un patron sin "/" solo casa en la raiz de data/ (``*.ctx*.json`` no debe
    alcanzar ``diag/x.ctx1.json``).
    """
    if "/" in pattern:
        return PurePosixPath(rel).match(pattern)
    return "/" not in rel and fnmatch.fnmatchcase(rel, pattern)


def _within(path: Path, root: Path) -> bool:
    """True si ``path`` resuelto cuelga de ``root`` resuelto."""
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except OSError:
        return False


def _iter_files(root: Path):
    """Ficheros bajo ``root``, sin seguir symlinks ni entrar en canonicos."""
    stack = [Path(root)]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir())
        except OSError as exc:
            print(f"[!] no se pudo leer {current}: {exc}", file=sys.stderr)
            continue
        for child in children:
            try:
                if child.is_symlink():
                    continue
                if child.is_dir():
                    if child.relative_to(root).as_posix() in CANONICAL_DIRS:
                        continue
                    stack.append(child)
                elif child.is_file():
                    yield child
            except OSError as exc:
                print(f"[!] no se pudo inspeccionar {child}: {exc}", file=sys.stderr)


def build_inventory(
    data_dir: Path,
    min_age_minutes: float = 10.0,
    now: Optional[float] = None,
) -> List[Dict]:
    """Inventario clasificado de ``data_dir`` segun ``POLICY`` y ``CANONICAL``.

    Cada entrada es un dict serializable (``path`` relativo posix, bytes, mtime,
    edad en minutos, regla casada y ``status``: keep / delete / skip_recent).
    """
    root = Path(data_dir)
    now = time.time() if now is None else now
    entries: List[Dict] = []
    groups: Dict[int, List[Dict]] = {}

    for path in _iter_files(root):
        try:
            st = path.stat()
        except OSError as exc:
            print(f"[!] no se pudo leer {path}: {exc}", file=sys.stderr)
            continue
        rel = path.relative_to(root).as_posix()
        age_minutes = max(0.0, (now - st.st_mtime) / 60.0)
        entry: Dict = {
            "path": rel,
            "bytes": st.st_size,
            "mtime": _iso(st.st_mtime),
            "mtime_epoch": st.st_mtime,
            "age_minutes": round(age_minutes, 2),
            "canonical": bool(_canonical_name(rel)),
            "pattern": None,
            "keep_latest": None,
            "status": "keep",
            "reason": "sin regla",
        }
        if entry["canonical"]:
            entry["reason"] = "canonico"
        else:
            for index, rule in enumerate(POLICY):
                if _matches(rel, rule.pattern):
                    entry["pattern"] = rule.pattern
                    entry["keep_latest"] = rule.keep_latest
                    entry["reason"] = rule.nota
                    groups.setdefault(index, []).append(entry)
                    break
        entries.append(entry)

    for index, group in groups.items():
        rule = POLICY[index]
        group.sort(key=lambda e: (-e["mtime_epoch"], e["path"]))
        for pos, entry in enumerate(group):
            if pos < rule.keep_latest:
                entry["status"] = "keep"
                entry["reason"] = (
                    f"{rule.nota}; dentro de keep_latest {rule.keep_latest} "
                    f"({pos + 1}/{len(group)})"
                )
            elif entry["age_minutes"] < min_age_minutes:
                entry["status"] = "skip_recent"
                entry["reason"] = (
                    f"reciente: {entry['age_minutes']:.1f} min < {min_age_minutes:g}"
                )
            else:
                entry["status"] = "delete"
                entry["reason"] = (
                    f"{rule.nota}; fuera de keep_latest {rule.keep_latest}"
                )

    entries.sort(key=lambda e: e["path"])
    return entries


def apply_actions(
    entries: Sequence[dict],
    data_dir: Path,
    backup_dir: Path | None = None,
    stream=None,
) -> dict:
    """Borra (o mueve a ``backup_dir``) las entradas con ``status == delete``.

    Revalida cada ruta: nunca sale de ``data_dir`` y nunca toca un canonico.
    Devuelve contadores (``deleted``, ``moved``, ``errors``, ``bytes``).
    """
    stream = sys.stdout if stream is None else stream
    root = Path(data_dir).resolve()
    backup_root = Path(backup_dir).resolve() if backup_dir is not None else None
    stats = {"deleted": 0, "moved": 0, "errors": 0, "bytes": 0}

    for entry in entries:
        if entry.get("status") != "delete":
            continue
        src = (root / entry["path"]).resolve()
        if not _within(src, root):
            entry["status"] = "error"
            entry["reason"] = "fuera de DATA_DIR: se rechaza"
            stats["errors"] += 1
            print(f"[X] fuera de DATA_DIR, no se toca: {entry['path']}", file=stream)
            continue
        if _canonical_name(entry["path"]):
            entry["status"] = "keep"
            entry["reason"] = "canonico"
            print(f"[!] canonico protegido: {entry['path']}", file=stream)
            continue
        if not src.is_file():
            entry["status"] = "error"
            entry["reason"] = "no es un fichero"
            stats["errors"] += 1
            print(f"[X] no es un fichero: {entry['path']}", file=stream)
            continue

        try:
            if backup_root is not None:
                dest = (backup_root / entry["path"]).resolve()
                if not _within(dest, backup_root):
                    raise OSError("destino de backup fuera de --backup-dir")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest = dest.with_name(f"{dest.name}.dup-{_ts()}")
                shutil.move(str(src), str(dest))
                entry["status"] = "moved"
                entry["moved_to"] = str(dest)
                entry["reason"] = f"movido a {dest}"
                stats["moved"] += 1
            else:
                src.unlink()
                entry["status"] = "deleted"
                stats["deleted"] += 1
            stats["bytes"] += int(entry.get("bytes") or 0)
        except OSError as exc:
            entry["status"] = "error"
            entry["reason"] = f"IO: {exc}"
            stats["errors"] += 1
            print(f"[X] fallo al procesar {entry['path']}: {exc}", file=stream)

    return stats


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m reii.diag data-hygiene",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help=f"raiz a inventariar (por defecto {DATA_DIR})",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="solo imprime el inventario (modo por defecto)",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="borra los candidatos (o los mueve si hay --backup-dir)",
    )
    ap.add_argument(
        "--min-age-minutes",
        type=float,
        default=10.0,
        metavar="MIN",
        help="no tocar ficheros modificados hace menos de MIN minutos (default 10)",
    )
    ap.add_argument(
        "--backup-dir",
        default=None,
        metavar="DIR",
        help="con --apply, mueve aqui preservando la ruta relativa en vez de borrar",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="inventario como JSON por stdout (avisos a stderr)",
    )
    return ap


def _print_inventory(entries: Sequence[Dict], out) -> None:
    width = max((len(e["path"]) for e in entries), default=0)
    for entry in entries:
        status = entry["status"]
        if status == "skip_recent":
            label = "[skip: reciente]"
        elif status == "moved":
            label = "[moved]"
        elif status in ("delete", "deleted"):
            label = "[del]"
        elif status == "error":
            label = "[X]"
        else:
            label = "[keep]"
        print(
            f"  {label:<16} {entry['path']:<{width}}  ({entry['reason']})",
            file=out,
        )


def run(argv=None) -> int:
    """Contrato del CLI: lo invoca ``reii.diag.cli`` con el argv restante."""
    utf8_stdout()
    args = build_parser().parse_args(argv)

    out = sys.stderr if args.json else sys.stdout
    root = Path(args.data_dir).expanduser()
    if not root.is_dir():
        print(f"[X] no existe el directorio {root}", file=out)
        return 1
    root = root.resolve()

    min_age = max(0.0, float(args.min_age_minutes))
    backup_dir = Path(args.backup_dir).expanduser() if args.backup_dir else None
    if backup_dir is not None and _within(backup_dir, root):
        print(
            f"[!] --backup-dir cuelga de --data-dir; el proximo inventario "
            f"volvera a ver esos ficheros: {backup_dir}",
            file=out,
        )

    entries = build_inventory(root, min_age_minutes=min_age)

    print("=" * 72, file=out)
    print(
        "HIGIENE DE data/  --  " + ("APPLY" if args.apply else "DRY-RUN"),
        file=out,
    )
    print("=" * 72, file=out)
    print(f"DATA_DIR: {root}", file=out)
    print(
        f"min-age: {min_age:g} min | backup: "
        f"{backup_dir if backup_dir is not None else '(ninguno, se borra)'}",
        file=out,
    )
    print("-" * 72, file=out)
    _print_inventory(entries, out)

    if args.apply:
        stats = apply_actions(
            entries,
            root,
            backup_dir=backup_dir,
            stream=out,
        )
        if backup_dir is not None:
            print("-" * 72, file=out)
            _print_inventory(
                [e for e in entries if e["status"] in ("moved", "error")],
                out,
            )
    else:
        stats = {"deleted": 0, "moved": 0, "errors": 0, "bytes": 0}

    candidates = [e for e in entries if e["status"] in ("delete", "deleted", "moved")]
    cand_bytes = sum(int(e.get("bytes") or 0) for e in candidates)
    recent = sum(1 for e in entries if e["status"] == "skip_recent")
    total_bytes = sum(int(e.get("bytes") or 0) for e in entries)

    if args.apply and backup_dir is not None:
        verb = "movidos a backup"
    elif args.apply:
        verb = "borrados"
    else:
        verb = "se borrarian (dry-run)"

    print("=" * 72, file=out)
    print(
        f"[ok] inventario: {len(entries)} fichero(s), {_human(total_bytes)} "
        f"({len(CANONICAL_DIRS)} dir(s) canonico(s) sin recorrer)",
        file=out,
    )
    print(
        f"[ok] {len(candidates)} fichero(s) / {_human(cand_bytes)} {verb}; "
        f"{recent} protegido(s) por edad",
        file=out,
    )

    if args.json:
        payload = {
            "data_dir": str(root),
            "generated_at": _iso(time.time()),
            "apply": bool(args.apply),
            "min_age_minutes": min_age,
            "backup_dir": str(backup_dir.resolve()) if backup_dir is not None else None,
            "policy": [rule._asdict() for rule in POLICY],
            "canonical": list(CANONICAL),
            "totals": {
                "files": len(entries),
                "bytes": total_bytes,
                "candidates": len(candidates),
                "candidate_bytes": cand_bytes,
                "skip_recent": recent,
                "deleted": stats["deleted"],
                "moved": stats["moved"],
                "errors": stats["errors"],
            },
            "files": entries,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 1 if stats["errors"] else 0


def main(argv=None) -> int:
    """Alias de ``run`` (paridad con ``db-hygiene``)."""
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
