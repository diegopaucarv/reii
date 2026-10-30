#!/usr/bin/env python
"""Preflight de artefactos REII (patch 22).

Decide si los parametros y la seleccion guardados corresponden al corpus que
hay en disco, y si el estudio de Optuna tiene artefactos conocidos. No
reimplementa el pipeline: lee los artefactos y delega en los scripts.

    python 6_preflight.py                        # informe; no toca nada
    python 6_preflight.py --study reii_probe_v6
    python 6_preflight.py --exec                 # ejecuta el plan si los gates pasan

Gates (bloquean --exec; exit 1):
    G1  el estudio del corpus actual no gasta presupuesto en centinelas
    G2  el ganador PERSISTIDO (data/best_params.json) no toca un parametro
        _BOUNDARY_EXPANDABLE

G2 se evalua sobre el ganador persistido, no sobre el "mejor 1er objetivo" de
cada etapa: el primero es la decision, el segundo es exploracion. Un trial que
maximiza cobertura y toca el piso `min_r2_threshold` no invalida la corrida si
la solucion guardada esta en el interior; se informa como borde exploratorio.

Los estudios se filtran por la huella del corpus (el nombre termina en
`_<fingerprint>`): los estudios de otro corpus no dicen nada del actual. Sin
ese filtro, un prefijo reutilizado (`reii_probe_v6`) arrastraria los estudios
viejos tras cambiar el corpus.

Sellos: si data/best_params.json no sella el corpus actual, el informe NO lo
trata como fallo sino como disparador de re-tuneo (con el presupuesto
pre-registrado, no con el generico de 200 trials de Config.optimize_trials).
Los gates sobre el estudio solo se evaluan cuando el sello coincide.

Exit: 0 listo · 1 gate fallido · 2 entorno · 3 hace falta --allow-retune.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
BEST = os.path.join(DATA, "best_params.json")
SELECTION = os.path.join(DATA, "method_selection.json")
SELECTION_LOG = os.path.join(DATA, "method_selection_log.jsonl")
MW = os.path.join(ROOT, "src", "reii", "main_workflow.py")

COMPOSE = ["docker", "compose", "--profile", "batch", "run", "--rm", "workflow"]

#: Un bloque de parametros del diagnostico imprime 11 claves; exigimos que al
#: menos estas coincidan para identificar el trial persistido. Menos que eso y
#: la coincidencia seria plausible por azar (11 claves, valores continuos).
MIN_SHARED_KEYS = 8

OK = "  [ok]  "
WARN = "  [!]   "
FAIL = "  [X]   "


# ── Huella del corpus (espejo de main_workflow._corpus_fingerprint) ──────────
def corpus_fingerprint() -> str:
    """sha256 de nombre + mtime + tamano de las entradas, recortado a 16.

    Es el MISMO calculo que ``main_workflow._corpus_fingerprint`` (L706).
    Se replica aqui para no importar el modulo (arrastra Stanza: 60-190 s).
    Si main_workflow cambia su definicion de corpus, esto hay que cambiarlo.
    """
    h = hashlib.sha256()
    paths = []
    paths += sorted(glob.glob(os.path.join(DATA, "txt_outputs", "tmp", "*.json")))
    paths += sorted(glob.glob(os.path.join(DATA, "txt_outputs", "*.txt")))
    meta = os.path.join(DATA, "txt_outputs", "Refined_Database.csv")
    if os.path.exists(meta):
        paths.append(meta)
    for p in paths:
        h.update(os.path.basename(p).encode("utf-8"))
        h.update(str(os.path.getmtime(p)).encode("utf-8"))
        h.update(str(os.path.getsize(p)).encode("utf-8"))
    return h.hexdigest()[:16]


# ── Politica de bordes, leida del fuente (no duplicada) ─────────────────────
def boundary_policy() -> dict:
    """Categorias del auditor de bordes tomadas de main_workflow.py.

    Se lee el texto en vez de importar (Stanza) y en vez de duplicar las
    listas (se desincronizarian en el primer patch que las toque).
    """
    empty = {"expandable": set(), "structural": set(), "declined": set()}
    try:
        with open(MW, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
    except OSError:
        return empty
    out = {}
    for key, pat, keyed in (
        ("expandable", r"_BOUNDARY_EXPANDABLE\s*=\s*\((.*?)\n\)", False),
        ("structural", r"_BOUNDARY_STRUCTURAL\s*=\s*\((.*?)\n\)", False),
        ("declined", r"_BOUNDARY_DECLINED\s*=\s*\{(.*?)\n\}", True),
    ):
        m = re.search(pat, src, re.S)
        body = m.group(1) if m else ""
        out[key] = set(
            re.findall(
                r'"([A-Za-z0-9_]+)"\s*:' if keyed else r'"([A-Za-z0-9_]+)"', body
            )
        )
    if not out["expandable"] and not out["structural"]:
        return empty
    return out


# ── Diagnostico (delegado en 3_diagnose_optuna_db.py) ───────────────────────
def run_diagnose(study: str):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "3_diagnose_optuna_db.py", "--study", study]
    try:
        p = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=300,
        )
    except Exception as e:  # noqa: BLE001 - el preflight informa, no cae
        return 2, "[X] no se pudo ejecutar el diagnostico: %s" % e
    return p.returncode, (p.stdout or "") + (p.stderr or "")


_RE_PARAM_LINE = re.compile(r"^\s+([A-Za-z0-9_]+)\s+(-?\d+(?:\.\d+)?)\s+(\S.*)$")
_RE_EDGE = "\u25c4"


def parse_diagnose(text: str) -> list:
    """Centinelas y bloque del ganador por estudio.

    Solo se guarda el bloque ``Trial #N (mejor 1er objetivo)``: los demas
    trials son exploracion, no la decision. Del bloque se conservan los
    valores (para poder emparejarlo con best_params.json) y que claves llevan
    la marca de borde.
    """
    studies, cur, in_block = [], None, False
    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^ESTUDIO \d+:\s*(\S+)$", line)
        if m:
            cur = {
                "name": m.group(1),
                "centinelas": None,
                "n_trials": None,
                "winner_trial": None,
                "values": {},
                "edges": [],
            }
            studies.append(cur)
            in_block = False
            continue
        if cur is None:
            continue
        m = re.search(r"Centinelas[^:]*:\s*(\d+)/(\d+)", line)
        if m:
            cur["centinelas"] = int(m.group(1))
            cur["n_trials"] = int(m.group(2))
            continue
        m = re.search(r"Trial #(\d+)\s*\(mejor 1er objetivo\)", line)
        if m:
            cur["winner_trial"] = int(m.group(1))
            in_block = True
            continue
        if not in_block:
            continue
        m = _RE_PARAM_LINE.match(line)
        if m:
            name, value, rest = m.group(1), float(m.group(2)), m.group(3)
            cur["values"][name] = value
            if _RE_EDGE in rest:
                cur["edges"].append((cur["winner_trial"], name))
            continue
        if line.strip():
            # fin del bloque: la siguiente seccion no trae valores numericos
            in_block = False
    return studies


def _same_number(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def persisted_blocks(studies: list, params: dict) -> list:
    """Bloques del diagnostico que corresponden al ganador persistido.

    Empareja por VALOR sobre las claves compartidas: best_params.json no
    guarda el id del trial, y el diagnostico solo imprime el bloque del
    ganador de cada etapa. Si ningun bloque coincide (p.ej. best_params.json
    escrito a mano por una sonda), el llamador cae a la union conservadora.
    """
    hits = []
    for s in studies:
        vals = s.get("values") or {}
        shared = [k for k in vals if isinstance(params.get(k), (int, float))]
        if len(shared) < MIN_SHARED_KEYS:
            continue
        if all(_same_number(float(params[k]), vals[k]) for k in shared):
            hits.append(s)
    return hits


# ── Utiles ──────────────────────────────────────────────────────────────────
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}


def sealed_fingerprint(doc):
    if not isinstance(doc, dict):
        return None
    ctx = doc.get("context")
    if isinstance(ctx, dict) and ctx.get("corpus_fingerprint"):
        return ctx["corpus_fingerprint"]
    return doc.get("corpus_fingerprint")


def param_dict(doc):
    if not isinstance(doc, dict):
        return {}
    params = doc.get("best_params") or doc.get("params") or {}
    return params if isinstance(params, dict) else {}


def running_workflows():
    cmd = [
        "docker",
        "ps",
        "--filter",
        "name=reii-workflow-run",
        "--format",
        "{{.Names}}",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        return []


def build_plan(args, retune):
    steps = []
    if retune:
        steps.append(
            (
                "re-tuneo (presupuesto pre-registrado)",
                COMPOSE
                + [
                    "python",
                    "-u",
                    "src/reii/main_workflow.py",
                    "--force",
                    "--optimize-only",
                    "--trials",
                    str(args.trials),
                    "--stage1-fraction",
                    str(args.stage1_fraction),
                    "--study-name",
                    args.study,
                    "--no-optuna-warm-start",
                ],
            )
        )
    steps.append(
        (
            "select-v4",
            COMPOSE
            + [
                "python",
                "-u",
                "src/reii/main_workflow.py",
                "--force",
                "--mode",
                "select",
            ],
        )
    )
    steps.append(
        (
            "auditoria estricta",
            COMPOSE + ["python", "-u", "src/9_selection_audit.py", "--strict"],
        )
    )
    return steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Preflight de artefactos REII: sellos, centinelas y bordes."
    )
    ap.add_argument("--study", default="reii_probe_v6", help="prefijo del estudio")
    ap.add_argument("--trials", type=int, default=40, help="presupuesto pre-registrado")
    ap.add_argument("--stage1-fraction", type=float, default=0.5)
    ap.add_argument("--max-centinelas", type=int, default=0)
    ap.add_argument(
        "--exec", action="store_true", help="ejecuta el plan si los gates pasan"
    )
    ap.add_argument(
        "--allow-busy", action="store_true", help="no abortar si hay un run vivo"
    )
    ap.add_argument(
        "--allow-retune",
        action="store_true",
        help="permite que --exec lance el re-tuneo (horas de maquina)",
    )
    ap.add_argument(
        "--keep-snapshots",
        action="store_true",
        help="no borrar data/method_selection*.json antes de select",
    )
    ap.add_argument(
        "--no-retune",
        action="store_true",
        help=argparse.SUPPRESS,  # interno: tras un re-tuneo no se re-tunea otra vez
    )
    ap.add_argument("--json", action="store_true", help="salida maquina-legible")
    args = ap.parse_args(argv)

    fp_now = corpus_fingerprint()
    policy = boundary_policy()
    failures = []
    report = {
        "fingerprint_now": fp_now,
        "best_params": {},
        "selection": {},
        "studies": [],
        "edges": {},
        "gates": {},
    }

    # ── sello de best_params ────────────────────────────────────────────
    params = {}
    if not os.path.exists(BEST):
        report["best_params"] = {"state": "MISSING", "fingerprint": None}
    else:
        b = load_json(BEST)
        if "_error" in b:
            report["best_params"] = {"state": "UNREADABLE", "error": b["_error"]}
            failures.append("best_params.json ilegible: %s" % b["_error"])
        else:
            fp_b = b.get("corpus_fingerprint")
            state = (
                "MATCH" if fp_b == fp_now else ("UNSEALED" if not fp_b else "MISMATCH")
            )
            params = param_dict(b)
            report["best_params"] = {
                "state": state,
                "fingerprint": fp_b,
                "score": b.get("score"),
                "n_params": len(params),
            }
    state = report["best_params"].get("state")
    retune = state != "MATCH" and not args.no_retune
    if state != "MATCH" and args.no_retune:
        failures.append("el re-tuneo no sello el corpus (best_params sigue %s)" % state)

    # ── sello de la seleccion (informativo: se regenera) ────────────────
    if os.path.exists(SELECTION):
        s = load_json(SELECTION)
        fp_s = sealed_fingerprint(s)
        method = None
        if isinstance(s, dict):
            method = s.get("selected_method") or s.get("method")
        report["selection"] = {
            "state": "MATCH" if fp_s == fp_now else "STALE",
            "fingerprint": fp_s,
            "method": method,
        }
    else:
        report["selection"] = {"state": "MISSING", "fingerprint": None}

    # ── gates sobre el estudio: solo con sello vigente ────────────────
    # Los estudios no se evaluan si el sello no coincide: sus parametros no
    # describen este corpus (y con --no-retune se aborta de todas formas).
    if state != "MATCH":
        report["studies"] = []
    else:
        rc, text = run_diagnose(args.study)
        studies = parse_diagnose(text) if "ESTUDIO" in text else []
        # Solo los estudios de ESTE corpus: el nombre lleva la huella al final
        # (`<prefijo>_stage1_wc_<fp>`). Un prefijo reutilizado arrastraria los
        # estudios del corpus anterior y los gates dirian cualquier cosa.
        studies = [s for s in studies if s["name"].endswith(fp_now)]
        report["studies"] = studies
        if not studies:
            failures.append(
                "G1/G2 sin estudios del corpus %s para el prefijo %r (rc=%d)"
                % (fp_now, args.study, rc)
            )
        total_cent = sum((s.get("centinelas") or 0) for s in studies)
        if total_cent > args.max_centinelas:
            failures.append(
                "G1 centinelas=%d > %d (presupuesto quemado en trials inviables)"
                % (total_cent, args.max_centinelas)
            )

        expandable = policy["expandable"]
        persisted = persisted_blocks(studies, params) if studies else []
        if persisted:
            censored = sorted(
                {p for s in persisted for (_, p) in s["edges"] if p in expandable}
            )
            source = "persistido"
        else:
            # Sin emparejamiento (best_params.json editado a mano, o el trial
            # persistido no es el "mejor 1er objetivo" de su etapa): se cae a
            # la union de todos los bloques. Es conservador a proposito.
            censored = sorted(
                {p for s in studies for (_, p) in s["edges"] if p in expandable}
            )
            source = "union(conservador)"
            if censored:
                failures.append(
                    "G2 no se pudo identificar el trial persistido; se aplica la "
                    "union de bordes ampliables (conservador): %s" % ", ".join(censored)
                )
        if censored and source == "persistido":
            failures.append(
                "G2 el ganador persistido toca un borde ampliable: %s"
                % ", ".join(censored)
            )

        def _hits(cat):
            return sorted(
                {
                    (s["name"], t, p)
                    for s in studies
                    for (t, p) in s["edges"]
                    if p in policy[cat]
                }
            )

        persisted_names = {p for s in persisted for (_, p) in s["edges"]}
        report["edges"] = {
            "expandable": [
                (n, t, p) for (n, t, p) in _hits("expandable") if p in censored
            ],
            "declined": list(_hits("declined")),
            "structural": list(_hits("structural")),
            "exploratory": [
                (n, t, p)
                for (n, t, p) in _hits("expandable")
                if p not in persisted_names
            ],
            "source": source,
        }

    report["gates"] = {
        "ok": not failures,
        "failures": failures,
        "retune": retune,
        "no_retune": bool(args.no_retune),
    }

    plan = build_plan(args, retune)
    report["plan"] = [{"step": t, "cmd": " ".join(c)} for t, c in plan]

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        # --json es para maquinas: nada de texto adicional en stdout.
        if not args.exec:
            return 0 if not failures else 1
    else:
        print("=" * 68)
        print("PREFLIGHT REII")
        print("=" * 68)
        print("huella corpus actual : %s" % fp_now)
        bp = report["best_params"]
        print(
            "best_params          : %-10s fp=%s  score=%s"
            % (bp.get("state"), bp.get("fingerprint"), bp.get("score"))
        )
        sl = report["selection"]
        print(
            "method_selection     : %-10s met=%s" % (sl.get("state"), sl.get("method"))
        )
        for s in report["studies"]:
            print(
                "estudio              : %s  centinelas=%s/%s"
                % (s["name"], s.get("centinelas"), s.get("n_trials"))
            )
        for s in report["studies"]:
            if s.get("values"):
                print(
                    "  ganador etapa       : trial #%s  (%d claves)"
                    % (s.get("winner_trial"), len(s["values"]))
                )
        for cat in ("expandable", "declined", "structural"):
            hits = report["edges"].get(cat) or []
            if hits:
                print(
                    "bordes %-12s : %s"
                    % (cat, ", ".join(sorted({p for _, _, p in hits})))
                )
        expl = report["edges"].get("exploratory") or []
        if expl:
            print(
                "bordes exploratorio  : %s  (trial no persistido)"
                % ", ".join(sorted({p for _, _, p in expl}))
            )
        if report["edges"].get("source"):
            print("fuente de los bordes : %s" % report["edges"]["source"])
        print("-" * 68)
        if retune:
            print(WARN + "best_params NO sella este corpus -> hace falta re-tuneo")
        for f in failures:
            print(FAIL + f)
        if not failures and not retune:
            print(OK + "los gates pasan")
        print("=" * 68)

    if not args.exec:
        if args.json:
            return 0 if not failures else 1
        print("\nplan sugerido:")
        for i, (title, cmd) in enumerate(plan, 1):
            print("  %d) %s\n     %s" % (i, title, " ".join(cmd)))
        print("\n(--exec para ejecutarlo; el re-tuneo exige ademas --allow-retune)")
        return 0 if not failures else 1

    if failures:
        print("\n[X] --exec abortado: los gates no pasan.")
        return 1
    busy = running_workflows()
    if busy and not args.allow_busy:
        print(
            "\n[X] hay un run vivo (%s); espera o usa --allow-busy." % ", ".join(busy)
        )
        return 1
    if retune and not args.allow_retune:
        print(
            "\n[!] el re-tuneo cuesta horas: repite con --allow-retune para lanzarlo."
        )
        print("    comando:\n     " + " ".join(plan[0][1]))
        return 3

    if retune:
        title, cmd = plan[0]
        print("\n>>> %s\n    %s" % (title, " ".join(cmd)))
        if subprocess.call(cmd, cwd=ROOT) != 0:
            print("\n[FAIL] el re-tuneo no termino bien; no se continua.")
            return 1
        print("\n[i] re-optimizacion terminada; volviendo a validar sellos y gates.")
        # --exec: tras el re-tuneo se corre select + auditoria sin preguntar
        # otra vez. --no-retune: si el sello sigue sin coincidir se aborta en
        # vez de re-tunear en bucle.
        argv2 = ["--study", str(args.study), "--exec", "--no-retune"]
        if args.json:
            argv2.append("--json")
        return main(argv2)

    if not args.keep_snapshots:
        for p in (SELECTION, SELECTION_LOG):
            if os.path.exists(p):
                os.remove(p)
                print("[i] snapshot stale borrado: %s" % os.path.relpath(p, ROOT))

    for i, (title, cmd) in enumerate(plan, 1):
        print("\n>>> [%d/%d] %s\n    %s" % (i, len(plan), title, " ".join(cmd)))
        if subprocess.call(cmd, cwd=ROOT) != 0:
            print("\n[X] el paso %d fallo; se detiene el plan." % i)
            return 1
    print("\n" + OK + "pipeline completo y auditado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
