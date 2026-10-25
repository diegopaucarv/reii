"""Utilidades compartidas por los scripts de diagnóstico 2_..5_ de REII.

Hechos del repo en los que se apoyan (verificados contra src/reii/main_workflow.py):
  * `reii` solo está instalado dentro de la imagen Docker (pip install -e .),
    y docker-compose monta ./src, ./data, ./models, ./output, ./ia, NO la raíz.
    => los comandos se lanzan con `docker compose --profile batch run --rm workflow ...`
       (igual que tu comando de rerun_classification), no con sys.executable del host.
  * main_workflow.py carga `uwu` al importarse y SALTA los documentos ya presentes
    en PostgreSQL salvo que se pase --force. Todo script debe pasar --force.
  * Flags CLI (añadidos por este parche): --trials, --seed, --study-name,
    --optimize-only, --bench, --bench-coref. Ya existían: --mode, --uce-target-size, --force.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data"
DIAG_DIR = DATA_DIR / "diag"
BEST_PARAMS = DATA_DIR / "best_params.json"  # == Optimizador._best_params_path()

# Espacio de búsqueda por defecto (get_search_space con min_forms_uc=[10,14], tsj=3)
DEFAULT_SPACE = {
    "min_forms_uc_1": (6, 14),
    "forms_gap": (2, 7),
    "tsj": (2, 5),
    "min_cluster_size_cdh": (0.05, 0.20),
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
