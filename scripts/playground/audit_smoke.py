#!/usr/bin/env python
"""Smoke del auditor de seleccion (`audit-v4`).

Corre los fixtures de `audit_fixtures/` contra el auditor y compara veredicto,
codigo de salida con `--strict` e invariantes declaradas en cada fixture.

Mientras el diff de `docs/design_v5/AUDIT_V4.md` NO este aplicado, el auditor
propuesto se extrae del bloque de codigo marcado con `<!-- audit-v4-source -->`
y se ejecuta ESE codigo: el smoke valida la propuesta, no una copia. Cuando el
diff este aplicado (el script expone `AUDIT_CONTRACT`) se usa el fichero real.

El historial, el snapshot y la info de `best_params.json` se inyectan en
memoria: no se escribe ningun fichero, no se toca `data/`, `output/` ni la BD,
y no se importa `reii.main_workflow` (tarda >4 min por Stanza). Corre en <1 s.

Uso
---
    python scripts/playground/audit_smoke.py
    python scripts/playground/audit_smoke.py -v     # traza de la extraccion

Salida: tablas ASCII con `->`, `[ok]`, `[!]`, `[X]`. Nada de box-drawing.
Sale != 0 si algun veredicto no coincide con el esperado del fixture.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
AUDIT = SRC / "reii" / "diag" / "selection_audit.py"
DOC = ROOT / "docs" / "design_v5" / "AUDIT_V4.md"
FIXTURES = Path(__file__).resolve().parent / "audit_fixtures"
DOC_MARKER = "<!-- audit-v4-source -->"


# ══════════════════════════════════════════════════════════════════════════
# 1. Carga del auditor
# ══════════════════════════════════════════════════════════════════════════
def extract_from_doc(path: Path = DOC) -> str:
    """Reconstruye el fichero propuesto desde el bloque diff del documento."""
    text = path.read_text(encoding="utf-8")
    idx = text.find(DOC_MARKER)
    if idx < 0:
        raise SystemExit(f"[X] {path} no contiene el marcador {DOC_MARKER}")
    lines = text[idx + len(DOC_MARKER) :].splitlines()
    fence = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("```")), None
    )
    if fence is None:
        raise SystemExit(f"[X] falta el bloque de codigo tras {DOC_MARKER}")
    out: List[str] = []
    for line in lines[fence + 1 :]:
        if line.strip().startswith("```"):
            break
        if line.startswith("+++"):
            continue  # cabecera del diff
        if line.startswith("+"):
            out.append(line[1:])
        elif line.strip() == "":
            out.append("")
    if not out or "def main(" not in "\n".join(out):
        raise SystemExit("[X] el bloque extraido no parece el auditor")
    return "\n".join(out) + "\n"


def load_engine(verbose: bool = False) -> Tuple[Dict[str, Any], str]:
    if AUDIT.exists() and "AUDIT_CONTRACT" in AUDIT.read_text(encoding="utf-8"):
        source, origin = AUDIT.read_text(encoding="utf-8"), "aplicado"
    else:
        source, origin = extract_from_doc(), "propuesto (AUDIT_V4.md)"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    namespace: Dict[str, Any] = {
        "__name__": "selection_audit_under_test",
        "__file__": f"<{origin}>",
    }
    exec(compile(source, f"<{origin}>", "exec"), namespace)
    if verbose:
        print(f"  [i] motor: {origin}, {len(source.splitlines())} lineas")
    return namespace, origin


# ══════════════════════════════════════════════════════════════════════════
# 2. Fixtures
# ══════════════════════════════════════════════════════════════════════════
def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def prepare(fixture: Dict[str, Any], rule: str) -> List[Dict[str, Any]]:
    """Copia el historial y, si el fixture declara `rule="current"`, alinea la
    `rule_version` con la del modulo para que el fixture valga en v3 y en v4."""
    records = _clone(fixture.get("history") or [])
    if fixture.get("rule") != "current":
        return records
    for record in records:
        record["rule_version"] = rule
        selection = record.get("selection")
        if isinstance(selection, dict):
            selection["rule_version"] = rule
    return records


def expected_for(fixture: Dict[str, Any], rule: str) -> Tuple[Dict[str, Any], str]:
    """Elige la expectativa segun la generacion de regla que corre el modulo."""
    declared = str(fixture.get("rule") or "current")
    if declared not in ("current", rule):
        stale = fixture.get("expect_stale_rule")
        if isinstance(stale, dict):
            return stale, f"{rule} (fixture escrito para {declared})"
    return fixture.get("expect") or {}, rule


def run_fixture(
    namespace: Dict[str, Any], fixture: Dict[str, Any], rule: str
) -> Tuple[int, Dict[str, Any]]:
    records = prepare(fixture, rule)
    snapshot = fixture.get("snapshot")
    best_params = fixture.get("best_params")
    # Inyeccion en memoria: el auditor real lee ficheros, el smoke no.
    namespace["read_history"] = lambda path: _clone(records)
    namespace["_load_snapshot"] = (
        (lambda path: _clone(snapshot)) if snapshot is not None else (lambda path: None)
    )
    namespace["_best_params_info"] = lambda path: (
        dict(best_params)
        if best_params
        else {"path": str(path), "exists": False, "mtime": None, "size": None}
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = namespace["main"](
            ["--history", "fixture.json", "--json", "-", "--strict", "--limit", "5"]
        )
    text = buffer.getvalue().strip()
    try:
        report = json.loads(text)
    except ValueError:
        raise SystemExit(
            "[X] el auditor no emitio JSON con --json -\n--- stdout ---\n" + text[:2000]
        )
    return int(code), report


def compare(
    expect: Dict[str, Any], returncode: int, report: Dict[str, Any]
) -> List[str]:
    problems: List[str] = []
    summary = report.get("summary") or {}
    want_verdict = expect.get("verdict")
    if want_verdict and summary.get("verdict") != want_verdict:
        problems.append(
            f"veredicto esperado {want_verdict} -> observado {summary.get('verdict')}"
        )
    want_exit = expect.get("exit_strict")
    if want_exit is not None and int(returncode) != int(want_exit):
        problems.append(f"exit --strict esperado {want_exit} -> observado {returncode}")
    records = report.get("records") or []
    status = {}
    if records:
        status = {c["id"]: c["status"] for c in (records[0].get("checks") or [])}
    for ident, want in (expect.get("invariants") or {}).items():
        got = status.get(ident, "(ausente)")
        if got != want:
            problems.append(f"{ident} esperado {want} -> observado {got}")
    blob = json.dumps(report, ensure_ascii=True)
    for needle in expect.get("must_contain") or []:
        if needle not in blob:
            problems.append(f"falta '{needle}' en el informe")
    return problems


# ══════════════════════════════════════════════════════════════════════════
# 3. Main
# ══════════════════════════════════════════════════════════════════════════
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    namespace, origin = load_engine(args.verbose)
    rule = str(namespace.get("RULE_VERSION", "?"))
    contract = str(namespace.get("AUDIT_CONTRACT", "(sin contrato)"))
    print(f"SMOKE auditor {contract} | motor: {origin} | regla vigente: {rule}")

    paths = sorted(FIXTURES.glob("*.json"))
    if not paths:
        print(f"[X] sin fixtures en {FIXTURES}")
        return 2

    header = (
        f"  {'fixture':30}  {'esperado':10}  {'observado':10}  {'exit':>4}  "
        f"{'estado':6}  rama"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    failures: List[str] = []
    for path in paths:
        try:
            fixture = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            failures.append(f"{path.name}: JSON invalido ({exc})")
            print(f"  {path.name:30}  {'-':10}  {'-':10}  {'-':>4}  [X]     json")
            continue
        expect, branch = expected_for(fixture, rule)
        try:
            returncode, report = run_fixture(namespace, fixture, rule)
        except SystemExit as exc:
            failures.append(f"{path.name}: {exc}")
            print(f"  {path.name:30}  {'-':10}  {'-':10}  {'-':>4}  [X]     crash")
            continue
        problems = compare(expect, returncode, report)
        observed = str((report.get("summary") or {}).get("verdict"))
        estado = "[X]" if problems else "[ok]"
        print(
            f"  {path.name:30}  {str(expect.get('verdict') or '-'):10}  "
            f"{observed:10}  {returncode:>4}  {estado:6}  {branch}"
        )
        for problem in problems:
            print(f"       -> {problem}")
        if problems:
            failures.append(f"{path.name}: " + "; ".join(problems))

    print("")
    if failures:
        print(f"[X] {len(failures)} fixture(s) no coinciden con lo declarado:")
        for item in failures:
            print(f"    -> {item}")
        return 1
    print(f"[ok] {len(paths)} fixtures: veredictos, exit y invariantes coinciden.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - un smoke debe decir QUE fallo
        traceback.print_exc()
        sys.exit(3)
