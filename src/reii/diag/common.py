"""Utilidades compartidas por los comandos de ``reii.diag``.

Hechos del repo en los que se apoyan (verificados contra src/reii/main_workflow.py):

  * ``reii`` está instalado dentro de la imagen Docker (``pip install -e .``) y
    docker-compose monta ``./src``, ``./data``, ``./models``, ``./output``,
    ``./ia`` (NO la raíz). Dentro del contenedor ``python -m reii.diag.cli``
    resuelve solo.
  * En el HOST no hay ``reii`` instalado: el punto de entrada es el shim
    ``diag.py`` de la raíz, que añade ``src/`` a ``sys.path``. Cualquier proceso
    hijo que se lance en el host debe usar ese shim (``python diag.py ...``) o
    ``-m reii.diag.cli`` con ``PYTHONPATH=src``.
  * ``main_workflow.py`` carga ``uwu`` al importarse y SALTA los documentos ya
    presentes en PostgreSQL salvo que se pase ``--force``. Todo comando que
    lance el pipeline debe pasar ``--force``.
  * Flags del pipeline que usan estos comandos: ``--trials``, ``--seed``,
    ``--study-name``, ``--optimize-only``, ``--stage1-fraction``,
    ``--no-optuna-warm-start``, ``--mode``, ``--uce-target-size``, ``--force``.

``REPO`` se resuelve subiendo por el árbol hasta el directorio que contiene
``pyproject.toml`` y ``src/reii``. Se puede forzar con la variable de entorno
``REII_REPO``. No se usa ``parents[N]`` a secas porque el número de niveles
cambia entre un árbol de fuentes y un ``site-packages``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

_PKG_MARKER = Path("src") / "reii"


def _find_repo() -> Path:
    """Raíz del repo (la que contiene ``pyproject.toml`` y ``src/reii``)."""
    env = os.environ.get("REII_REPO")
    if env:
        candidate = Path(env).expanduser()
        if (candidate / "pyproject.toml").is_file() or (
            candidate / _PKG_MARKER
        ).is_dir():
            return candidate.resolve()
        print(
            f"[!] REII_REPO={env!r} no parece la raíz del repo; se ignora",
            file=sys.stderr,
        )

    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for base in (here.parent, *here.parents, cwd, *cwd.parents):
        if (base / "pyproject.toml").is_file() and (base / _PKG_MARKER).is_dir():
            return base
    # Último recurso: .../src/reii/diag/common.py -> repo. Sirve en el árbol de
    # fuentes; en un wheel instalado se queda corto y el llamador verá rutas
    # inexistentes (mejor que reventar al importar).
    parents = here.parents
    return parents[3] if len(parents) > 3 else cwd


REPO = _find_repo()
DATA_DIR = REPO / "data"
DIAG_DIR = DATA_DIR / "diag"
BEST_PARAMS = DATA_DIR / "best_params.json"  # == Optimizador._best_params_path()

# Espacio de búsqueda por defecto (get_search_space con min_forms_uc=[10,14], tsj=3)
DEFAULT_SPACE = {
    "min_forms_uc_1": (6, 14),
    "forms_gap": (2, 7),
    "tsj": (2, 5),
    # Espacio actual: K objetivo -> frac = 1/(2K). Se mantiene la clave legacy
    # (min_cluster_size_cdh) para re-analizar best_params.json antiguos.
    "cdh_target_k": (3, 20),
    "min_cluster_size_cdh": (0.025, 0.1667),
    "swap_iterations": (1, 5),
    "min_r2_threshold": (0.03, 0.10),
    "similarity_threshold": (0.25, 0.65),
    "coref_weight": (0.0, 0.4),
    "uc_window_size": (1, 5),
    "coref_context_units_tight": (1, 3),
    "coref_context_units_loose": (3, 5),
    "hdbscan_uce_min_cluster_frac": (0.01, 0.04),
    "hdbscan_uce_min_cluster_frac_loose": (0.005, 0.03),
}
SENTINEL = (-2.0, -1.0, 0.0)  # main_workflow._SENTINEL_TRIAL


def utf8_stdout() -> None:
    """Consola cp1252 de Windows: evita UnicodeEncodeError sin perder salida."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def detect_runner(preferred: Optional[str] = None) -> str:
    if preferred in ("docker", "local"):
        return preferred
    if shutil.which("docker") and (REPO / "docker-compose.yml").exists():
        return "docker"
    return "local"


def workflow_cmd(runner: str, *flags: str) -> List[str]:
    """Comando para ejecutar main_workflow.py con los flags dados."""
    py = ["python", "-u", "src/reii/main_workflow.py", *flags]
    if runner == "docker":
        return [
            "docker",
            "compose",
            "--profile",
            "batch",
            "run",
            "--rm",
            "workflow",
            *py,
        ]
    return [sys.executable, "-u", *py[2:]]


def diag_cmd(runner: str, *argv: str) -> List[str]:
    """Comando para ejecutar otro comando de ``reii.diag`` en el mismo entorno.

    En docker ``reii`` está instalado: ``python -m reii.diag.cli``. En el host
    se usa el shim de la raíz, que no requiere instalar el paquete.
    """
    if runner == "docker":
        return [
            "docker",
            "compose",
            "--profile",
            "batch",
            "run",
            "--rm",
            "workflow",
            "python",
            "-u",
            "-m",
            "reii.diag.cli",
            *argv,
        ]
    return [sys.executable, "-u", str(REPO / "diag.py"), *argv]


def run_to_log(
    cmd: List[str], log_path: Path, timeout: Optional[int], dry_run: bool
) -> Optional[str]:
    """Ejecuta cmd volcando stdout+stderr a log_path; devuelve el texto o None si falla."""
    print("  $ " + " ".join(cmd))
    if dry_run:
        return ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        try:
            rc = subprocess.run(
                cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            print(f"  ✗ timeout ({timeout}s); log parcial en {log_path}")
            return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if rc != 0:
        print(f"  ✗ returncode={rc}; cola del log ({log_path}):")
        print("    " + "\n    ".join(text.splitlines()[-8:]))
        return None
    return text


def parse_optimizer_results(text: str) -> List[Dict]:
    """Todas las líneas `[OPTIMIZER-RESULT] {json}` (una por etapa)."""
    out = []
    for m in re.finditer(r"^\[OPTIMIZER-RESULT\] (\{.*\})\s*$", text, re.M):
        try:
            out.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    return out


def parse_bench(text: str) -> Dict[str, float]:
    res = {}
    for k, v in re.findall(r"^\[BENCH\] (\w+)=([-\d.]+)\s*$", text, re.M):
        res[k] = float(v)
    return res


def at_boundary(name: str, value, space=DEFAULT_SPACE, tol=1e-9) -> bool:
    lo, hi = space.get(name, (None, None))
    if lo is None:
        return False
    return abs(value - lo) <= tol or abs(value - hi) <= tol
