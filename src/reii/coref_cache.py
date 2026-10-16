"""Cache en disco de cadenas coref precomputadas por context_units.

Módulo puro (sin imports pesados) para poder testear la serialización,
validación y escritura atómica sin importar main_workflow (que tarda ~281s
por torch+stanza+spacy+DB).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if os.name == "nt":
    import msvcrt
else:
    import fcntl

SCHEMA_VERSION: int = 1


def serialize_chains(chains: Dict[int, Dict[str, List]]) -> Dict[str, Any]:
    """Convierte {ctx:int -> {doc_id -> [chains]}} a dict JSON-safe
    (claves ctx como str). No muta la entrada."""
    return {str(ctx): by_doc for ctx, by_doc in chains.items()}


def deserialize_chains(raw: Any) -> Dict[int, Dict[str, List]]:
    """Inverso de serialize_chains. Convierte claves ctx de str a int.
    Lanza ValueError si la estructura no es válida."""
    if not isinstance(raw, dict):
        raise ValueError("chains debe ser un dict")
    out: Dict[int, Dict[str, List]] = {}
    for ctx_str, by_doc in raw.items():
        try:
            ctx = int(ctx_str)
        except (TypeError, ValueError) as e:
            raise ValueError(f"ctx inválido: {ctx_str!r}") from e
        if not isinstance(by_doc, dict):
            raise ValueError(f"chains[{ctx_str}] debe ser un dict")
        out[ctx] = {str(doc_id): list(chains) for doc_id, chains in by_doc.items()}
    return out


def build_cache_document(
    cache_key: str, fingerprint: str, chains: Dict[int, Dict[str, List]]
) -> Dict[str, Any]:
    """Construye el documento completo (metadata + chains serializadas)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "cache_key": cache_key,
        "corpus_fingerprint": fingerprint,
        "chains": serialize_chains(chains),
    }


def validate_cache_document(doc: Any, cache_key: str) -> bool:
    """True si doc es válido (schema_version correcto, cache_key coincide,
    chains deserializables). Nunca lanza; devuelve False ante cualquier
    anomalía."""
    try:
        if not isinstance(doc, dict):
            return False
        if doc.get("schema_version") != SCHEMA_VERSION:
            return False
        if doc.get("cache_key") != cache_key:
            return False
        deserialize_chains(doc.get("chains"))
        return True
    except (ValueError, TypeError, KeyError):
        return False


def atomic_write_json(path: str, obj: Any) -> None:
    """Escribe obj a path de forma atómica: tempfile en el mismo directorio
    + os.replace. Crea el directorio padre si no existe."""
    target = Path(path)
    parent = target.parent
    os.makedirs(parent or Path("."), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(parent or Path(".")), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_cache(path: str, cache_key: str) -> Optional[Dict[int, Dict[str, List]]]:
    """Carga y valida el cache. Devuelve None si el archivo no existe, es
    ilegible, está corrupto o su cache_key no coincide. Nunca lanza."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not validate_cache_document(doc, cache_key):
        return None
    try:
        return deserialize_chains(doc.get("chains"))
    except (ValueError, TypeError):
        return None


def missing_ctx_values(
    requested: Iterable[int], present: Dict[int, Dict[str, List]]
) -> List[int]:
    """Devuelve los ctx de `requested` que no están en `present`, ordenados
    y sin duplicados."""
    seen = set()
    out: List[int] = []
    for c in requested:
        c = int(c)
        if c not in present and c not in seen:
            seen.add(c)
            out.append(c)
    return sorted(out)


@contextlib.contextmanager
def file_lock(lock_path: str, timeout: float = 300.0, poll_interval: float = 0.25):
    """Lock de archivo cross-platform (fcntl.flock en POSIX, msvcrt.locking
    en Windows). El SO libera el lock automáticamente si el proceso muere,
    así que no hay locks huérfanos. Si no se puede adquirir en `timeout`
    segundos, lanza TimeoutError."""
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _acquire_file_lock(fd, timeout, poll_interval)
        yield
    finally:
        _release_file_lock(fd)
        os.close(fd)


def _acquire_file_lock(fd: int, timeout: float, poll_interval: float) -> None:
    start = time.monotonic()
    while True:
        try:
            if os.name == "nt":
                # msvcrt.locking necesita al menos 1 byte en el archivo
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() - start >= timeout:
                raise TimeoutError(
                    f"No se pudo adquirir el lock de archivo en {timeout:.0f}s"
                )
            time.sleep(poll_interval)


def _release_file_lock(fd: int) -> None:
    try:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
