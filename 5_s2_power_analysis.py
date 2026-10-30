#!/usr/bin/env python3
"""
Script 5 (OPCIONAL) - Potencia estadistica S2 vs S1 sobre el benchmark MOCK.

Aclaracion importante: tests/reii_optimizer_report.py NO mide tu corpus. Simula el
TPESampler de Optuna sobre paisajes sinteticos para comparar dos estrategias de
busqueda (S1 = fix de lookup; S2 = + factibilidad exacta por pares). Sirve para decidir
una mejora de INGENIERIA del optimizador; no explica por que tus clases salen mal.

Esta revision:
  * Itera TODOS los paneles del JSON (autodescubiertos por el tag antes del primer
    "|"), no solo "T=coste90_con_fix". Por cada panel y escenario calcula el delta
    pareado S2-S1, su IC95 y wins/n.
  * --min-n (8 por defecto): con n < min_n no declara significativo ni "sin
    evidencia"; imprime "muestra insuficiente (n=X < min_n=Y)" y la recomendacion del
    panel queda "indeterminado".
  * Recomendacion POR PANEL, mas una sintesis honesta: S2 solo importa si gana en los
    paneles de igualdad de TIEMPO (T=*); en A90 el presupuesto de trials es el mismo y
    S2 solo cambia que trials consume.
  * Streaming: subprocess.Popen con stdout=PIPE, stderr=STDOUT, bufsize=1, text=True,
    encoding="utf-8", errors="replace"; cada linea va a stdout y a
    data/diag/s2_bench.log. Antes capture_output=True dejaba 26-105 min de silencio y
    parecia colgado.
  * Se mantienen --reuse-json (acepta .md o .json), --bench-out, --out, --timeout,
    --seeds y --scenarios. Rutas resueltas a absolutas. Consola ASCII pura.
  * --dry-run imprime el comando del hijo y sale 0 sin ejecutarlo.
  * Si el hijo falla se muestra la cola de su salida (stdout y stderr fusionados) y se
    devuelve 1. El JSON resumen tiene la forma
    {panel: {escenario: {delta, ci, wins, n, significant}}}.

Contrato con el benchmark (verificado contra tests/reii_optimizer_report.py):
  * Flags: --workdir --tests --seeds --skip-tests --skip-bench --scenarios --out
    --check-patch --repo --base.
  * JSON: os.path.splitext(args.out)[0] + ".json", estructura
    {"tests_ok": bool, "benchmark": {escenario: {clave: [trial, ...]}}}.
  * Cada trial es (u/bk, coste, n_trials, n_validos_s2): r[0] = utilidad relativa al
    optimo. Como ambas estrategias usan las mismas semillas en el mismo orden, la
    diferencia es pareada.
  * Claves: tag y nombre unidos con "|", p.ej. "A90|S1 fix lookup" o
    "T=coste90_con_fix|S2 +factible exacto (pares)". El tag se autodescubre; los
    nombres de estrategia son "S1 fix lookup" y "S2 +factible exacto (pares)".

Tiempo: ~65 s por paisaje y escenario -> 32 semillas x 3 escenarios ~= 105 min.
Un --seeds menor basta para una primera pasada: --seeds 8 (~26 min) o --seeds 1 con
un solo escenario (~65 s), que es lo minimo para verificar el pipeline y el JSON.

Uso:
  python 5_s2_power_analysis.py --seeds 32 --scenarios 30:0.7,100:0.7,30:0.3
  python 5_s2_power_analysis.py --seeds 8                       # primera pasada
  python 5_s2_power_analysis.py --reuse-json bench_output.json  # solo analizar
  python 5_s2_power_analysis.py --dry-run                       # ver comando y salir
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent

# Nombres EXACTOS de las estrategias dentro de cada panel (el tag es lo que cambia
# entre paneles y se autodescubre).
NAME_REF = "S1 fix lookup"
NAME_S2 = "S2 +factible exacto (pares)"

SEP = "|"
LOG_PATH = REPO / "data" / "diag" / "s2_bench.log"


def _utf8_stdout():
    """Evita UnicodeEncodeError en consolas cp1252 (Windows)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def build_cmd(script, seeds, scenarios, out_md):
    """Comando del benchmark; out_md debe ser absoluto (el hijo corre con cwd=REPO)."""
    return [
        sys.executable,
        str(script),
        "--workdir",
        str(REPO),
        "--skip-tests",
        "--seeds",
        str(seeds),
        "--scenarios",
        scenarios,
        "--out",
        str(out_md),
    ]


def run_benchmark(script, seeds, scenarios, out_md, timeout):
    """Lanza el benchmark con streaming y devuelve la ruta del .json (o None)."""
    out_md = Path(out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(script, seeds, scenarios, out_md)
    print("-> " + " ".join(cmd), flush=True)
    print(f"-> log en streaming: {LOG_PATH}", flush=True)

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as e:
        print(f"ERROR: no se pudo lanzar el benchmark: {e}")
        return None

    timed_out = {"hit": False}

    def _kill():
        timed_out["hit"] = True
        try:
            proc.kill()
        except OSError:
            pass

    timer = threading.Timer(timeout, _kill)
    timer.daemon = True
    timer.start()

    tail = deque(maxlen=40)
    rc = None
    try:
        with open(LOG_PATH, "w", encoding="utf-8", errors="replace") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
                tail.append(line)
            rc = proc.wait()
    finally:
        timer.cancel()
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass

    if timed_out["hit"]:
        print(f"\nERROR: el benchmark supero --timeout ({timeout}s)")
        _print_tail(tail)
        return None
    if rc != 0:
        print(f"\nERROR: el benchmark fallo (returncode={rc})")
        # stderr se fusiono en stdout (stderr=STDOUT): la cola cubre ambos.
        _print_tail(tail, label="cola de salida (stdout y stderr fusionados)")
        print(f"-> log completo: {LOG_PATH}")
        return None
    return out_md.with_suffix(".json")


def _print_tail(tail, label="cola de salida"):
    if tail:
        print(f"--- {label} ---")
        sys.stdout.write("".join(tail))
        if not tail[-1].endswith("\n"):
            print()
    else:
        print("(el hijo no produjo ninguna linea)")


def discover_panels(data):
    """{panel: {escenario: {nombre: [trials]}}} autodescubierto del JSON.

    El panel es el tag antes del primer "|" de cada clave. No se hardcodea ninguno.
    """
    panels = {}
    for sc, bd in data.items():
        if not isinstance(bd, dict):
            continue
        for key, trials in bd.items():
            if SEP not in key:
                continue
            tag, name = key.split(SEP, 1)
            panels.setdefault(tag, {}).setdefault(sc, {})[name] = trials
    return panels


def paired_stats(ref, s2, min_n):
    """Delta pareado S2-S1, IC95, wins y n.

    Devuelve (stats, aviso, too_small). Con n < min_n, significant=False.
    """
    n = min(len(ref), len(s2))
    if n == 0:
        return None, "listas vacias", False
    # r[0] = u/bk (utilidad relativa al optimo); pareado por semilla
    d = np.array([r[0] for r in s2[:n]], dtype=float) - np.array(
        [r[0] for r in ref[:n]], dtype=float
    )
    if n > 1:
        ci = float(1.96 * d.std(ddof=1) / np.sqrt(n))
    else:
        ci = float("nan")
    too_small = n < min_n or n < 2
    if too_small:
        sig = False
    else:
        sig = bool(n > 1 and math.isfinite(ci) and (d.mean() - ci) > 0)
    stats = {
        "delta": float(d.mean()),
        "ci": ci if math.isfinite(ci) else None,
        "wins": int((d > 0).sum()),
        "n": int(n),
        "significant": sig,
    }
    note = None
    if len(s2) != len(ref):
        note = f"AVISO: listas de distinto tamano ({len(s2)} vs {len(ref)}); uso n={n}"
    return stats, note, too_small


def fmt_ci(ci):
    return "nan" if ci is None else f"{ci:.4f}"


def panel_verdict(panel, stats_by_sc, min_n):
    """Recomendacion legible por panel, honesta con muestras pequenas."""
    usable = [
        (sc, s)
        for sc, s in stats_by_sc.items()
        if s["n"] >= min_n and s["ci"] is not None
    ]
    if not usable:
        ns = ", ".join(
            f"{sc}: n={s['n']}" + ("" if s["ci"] is not None else " sin IC95")
            for sc, s in stats_by_sc.items()
        )
        return f"indeterminado (muestra insuficiente; {ns})"
    wins = [sc for sc, s in usable if s["significant"] and s["delta"] > 0]
    losses = [sc for sc, s in usable if s["significant"] and s["delta"] < 0]
    is_time = panel.startswith("T=")
    if wins:
        base = "S2 gana (IC95 > 0) en " + ", ".join(wins)
    elif losses:
        base = "S1 gana en " + ", ".join(losses)
    else:
        base = "sin evidencia (IC95 cruza 0)"
    if not is_time:
        base += " [mismo numero de trials: S2 solo cambia que trials consume]"
    return base


def main() -> int:
    _utf8_stdout()
    ap = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--bench-script", default=str(REPO / "tests" / "reii_optimizer_report.py")
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=32,
        help="paisajes sinteticos por escenario (32 ~= 105 min; 8 primera pasada)",
    )
    ap.add_argument(
        "--scenarios",
        default="30:0.7,100:0.7,30:0.3",
        help="lista P:rho separada por comas (coste pase coref : correlacion etapa1-2)",
    )
    ap.add_argument("--out", default="s2_results.json", help="JSON resumen de salida")
    ap.add_argument(
        "--bench-out",
        default="bench_output.md",
        help=".md que escribe el benchmark (su .json se escribe al lado)",
    )
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument(
        "--reuse-json",
        default=None,
        help="analiza un .json ya generado (acepta tambien el .md)",
    )
    ap.add_argument(
        "--min-n",
        type=int,
        default=8,
        help="minimo de semillas pareadas para declarar significancia (def. 8)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="imprime el comando del hijo y sale 0 sin ejecutarlo",
    )
    a = ap.parse_args()

    if a.min_n < 1:
        print(f"ERROR: --min-n debe ser >= 1 (recibido {a.min_n})")
        return 1

    bench_script = Path(a.bench_script).resolve()
    bench_out = Path(a.bench_out).resolve()

    if a.reuse_json:
        js = Path(a.reuse_json).resolve()
        if js.suffix.lower() != ".json":
            js = js.with_suffix(".json")  # acepta tambien el --out .md
        if a.dry_run:
            print(f"-> (dry-run) analizaria {js}; no se ejecuta nada")
            return 0
        if not js.exists():
            print(f"AVISO: no existe {js}")
            print("-> con --reuse-json no se relanza el benchmark; nada que analizar")
            return 0
        print(f"-> reutilizando {js}")
    else:
        if not bench_script.exists():
            print(f"ERROR: no existe {bench_script}")
            return 1
        if a.dry_run:
            cmd = build_cmd(bench_script, a.seeds, a.scenarios, bench_out)
            print("-> (dry-run) comando del hijo:")
            print("   " + " ".join(cmd))
            print("-> no se ejecuta; saliendo con 0")
            return 0
        js = run_benchmark(bench_script, a.seeds, a.scenarios, bench_out, a.timeout)

    if not js or not js.exists():
        print("ERROR: el benchmark no dejo ningun JSON de resultados")
        return 1
    try:
        raw = json.loads(js.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"ERROR: no se pudo leer {js}: {e}")
        return 1
    data = raw.get("benchmark", {})
    if not data:
        print(f"ERROR: {js} no trae seccion 'benchmark'")
        return 1

    panels = discover_panels(data)
    if not panels:
        print(f"ERROR: {js} no tiene claves con formato 'tag|estrategia'")
        return 1

    print(f"-> paneles detectados: {', '.join(panels)}")
    print(f"-> min_n={a.min_n}")

    res = {}
    for panel, by_sc in panels.items():
        print(f"\n=== PANEL {panel} ===")
        res[panel] = {}
        for sc, names in by_sc.items():
            ref = names.get(NAME_REF)
            s2 = names.get(NAME_S2)
            if ref is None or s2 is None:
                avail = ", ".join(repr(k) for k in names)
                print(f"{sc}: faltan claves; disponibles: {avail}")
                continue
            stats, note, too_small = paired_stats(ref, s2, a.min_n)
            if note:
                print(f"{sc}: {note}")
            if stats is None:
                print(f"{sc}: {note or 'sin datos'}, se omite")
                continue
            print(
                f"{sc}: delta(S2-S1)={stats['delta']:+.4f} +- {fmt_ci(stats['ci'])}"
                f"  gana {stats['wins']}/{stats['n']}  n={stats['n']}"
            )
            if too_small:
                if stats["n"] < a.min_n:
                    print(
                        f"{sc}: muestra insuficiente (n={stats['n']} < min_n={a.min_n})"
                        f" -> no se declara significativo"
                    )
                else:
                    print(
                        f"{sc}: muestra insuficiente para IC95 (n={stats['n']} < 2)"
                        f" -> no se declara significativo"
                    )
            else:
                print(f"{sc}: significativo={stats['significant']}")
            res[panel][sc] = stats

    # Paneles sin ningun escenario analizable fuera del JSON de salida
    res = {p: v for p, v in res.items() if v}
    if not res:
        print("ERROR: ningun panel/escenario analizable (revisa las claves listadas)")
        return 1

    out_path = Path(a.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nResumen escrito en {out_path}")

    print("\nRECOMENDACION POR PANEL:")
    verdicts = {}
    for panel, by_sc in res.items():
        verdicts[panel] = panel_verdict(panel, by_sc, a.min_n)
        print(f"  {panel}: {verdicts[panel]}")

    print("\nSINTESIS:")
    print("  S2 solo importa si gana en el panel de igualdad de TIEMPO (T=*).")
    print(
        "  En A90 el presupuesto de trials es el mismo y S2 solo cambia que"
        " trials consume."
    )
    time_panels = [p for p in res if p.startswith("T=")]
    if not time_panels:
        print("  VEREDICTO: indeterminado (el JSON no trae paneles T=* de tiempo).")
    else:
        usable = [
            (p, sc, s)
            for p in time_panels
            for sc, s in res[p].items()
            if s["n"] >= a.min_n and s["ci"] is not None
        ]
        if not usable:
            print(
                "  VEREDICTO: indeterminado (muestra insuficiente en los paneles"
                f" T=*: ninguno tiene n >= {a.min_n} con IC95 valido)."
            )
        elif any(s["significant"] and s["delta"] > 0 for _, _, s in usable):
            print("  VEREDICTO: S2 gana a igualdad de tiempo -> candidato a adoptar.")
        else:
            print("  VEREDICTO: sin evidencia a igualdad de tiempo -> queda S1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
