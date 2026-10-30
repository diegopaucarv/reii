#!/usr/bin/env python3
"""
Script 4 - Forma y coste de la segmentacion segun uce_target_size.

Que mide (verificado contra src/reii/main_workflow.py y data/diag/bench_uce*.log):

  main_workflow.py NO tiene --bench ni --bench-coref ni instrumentacion por etapa
  (seg_seconds / lemmatize_seconds / coref_seconds no existen). Lo unico medible sin
  parchearlo son dos lineas del log:

      [CLI] uce_target_size=26 (--uce-target-size override; ...)
      600 UCEs en 29 documentos. Promedio real: 23.9 palabras/UCE (objetivo uce_target_size=26).

  parse_uce_lines() las lee. El coste es el wall-clock entre el primer y el ultimo
  timestamp del log. Una corrida truncada se marca PARCIAL: cuenta para n_UCE (la
  segmentacion ocurre al principio) pero NUNCA para el coste.

Que cambio respecto a la version anterior:

  1. QUE SE CORRE != QUE SE AJUSTA. El ajuste (pendientes, extrapolacion, tabla) usa
     TODAS las filas de los logs bench_uce*.log descubiertos por glob en data/diag,
     no solo las de --sizes. --sizes limita UNICAMENTE lo que se lanza. La tabla lleva
     columna "origen" (log = leido de un log existente; ahora = recien ejecutado).
  2. --force-run ya no es un footgun: solo relanza los --sizes pedidos (~2 h por
     tamano), nunca "todos los tamanos".
  3. --target N evalua un target hipotetico SIN lanzar nada: predice n_UCE y
     pal_cr/UCE (modelo hiperbolico n_UCE*target ~ cte y modelo lineal), el target
     necesario para 376 UCE, y el coste estimado a partir de s/UCE. Marca si el
     target queda dentro/fuera del rango de diseno [14, 30].
  4. Coste normalizado: segundos por UCE (media observada y regresion
     wall_seconds ~ n_uces) y proyeccion de coste para un corpus mayor.
  5. Bloque de lectura con la aritmetica explicita: target "unit-fiel" (~28.6),
     rango de diseno [14, 30], aritmetica de COBERTURA de ALCESTE (376*26 ~ 68% del
     corpus -> faltan ~4555 tokens) y aviso de que los dos estimadores NO son
     independientes (misma identidad, corpus de palabras fijas).
  5b. Unidades separadas: "palabras crudas/UCE" (pal_cr/UCE, lo que mide el log)
     frente a "palabras analizadas/UCE" (13.95 de ALCESTE). No se comparan 23.9
     crudas con 13.95 analizadas sin decirlo.
  6. Salida ASCII pura en consola ([ok], [i], [!], [X], ->, >=, ~) y _utf8_stdout():
     la consola cp1252 no tiene los glifos de check/flecha y la version anterior
     reventaba con UnicodeEncodeError.

Uso:
  python 4_measure_uce_target_size.py --parse-only
  python 4_measure_uce_target_size.py --parse-only --extrapolate 5809
  python 4_measure_uce_target_size.py --parse-only --target 34
  python 4_measure_uce_target_size.py --force-run --sizes 34     # docker, ~2 h
  python 4_measure_uce_target_size.py --dry-run --sizes 34
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime

from diag_common import DIAG_DIR, detect_runner, run_to_log, workflow_cmd

ALCESTE_REF = {"uce_size": 26, "n_uce": 376, "words_per_uce_analyzed": 13.95}
# Copiado de main_workflow.calibrar_uce_target_size: cfg.uce_calibration_bounds.
UCE_BOUNDS = (14, 30)
LAUNCH_DEFAULT = (18, 22, 26, 30)

# Nombres de campos [BENCH] que NO existen hoy; se mantienen por si se instrumenta.
TIME_KEYS = ("seg_seconds", "lemmatize_seconds", "coref_seconds")
SHORT_TIME = {
    "seg_seconds": "seg_s",
    "lemmatize_seconds": "lem_s",
    "coref_seconds": "coref_s",
}
COMPLETE_MARKERS = ("Pipeline completado", "Dashboard keys escritas")

_RE_CLI_TARGET = re.compile(r"\[CLI\]\s+uce_target_size=(\d+)")
_RE_UCE_SUMMARY = re.compile(
    r"(\d+)\s+UCEs\s+en\s+(\d+)\s+documentos\.\s+"
    r"Promedio real:\s+([0-9.]+)\s+palabras/UCE\s+"
    r"\(objetivo uce_target_size=(\d+)\)"
)
_RE_BENCH = re.compile(r"^\[BENCH\] (\w+)=([-\d.]+)\s*$", re.M)
_RE_TS = re.compile(r"^\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})", re.M)
_RE_LOG_NAME = re.compile(r"bench_uce(\d+)\.log$")


def _utf8_stdout():
    """cp1252 no entiende todos los glifos; si se puede, emitir UTF-8."""
    out = getattr(sys, "stdout", None)
    if out is not None and hasattr(out, "reconfigure"):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def parse_uce_lines(text):
    """size/n_uces/n_docs/mean_words_per_uce a partir del log de main_workflow.py.

    Usa la ultima aparicion de cada linea (la segmentacion real va despues de la
    calibracion). Las lineas [BENCH] reales, si algun dia se instrumenta el
    workflow, solo anaden/sobrescriben campos *_seconds; nunca son obligatorias.
    """
    out = {}
    cli = _RE_CLI_TARGET.findall(text)
    if cli:
        out["size"] = int(cli[-1])
    summary = _RE_UCE_SUMMARY.findall(text)
    if summary:
        n_uces, n_docs, mean, target = summary[-1]
        out["n_uces"] = int(n_uces)
        out["n_docs"] = int(n_docs)
        out["mean_words_per_uce"] = float(mean)
        out.setdefault("size", int(target))
    for key, value in _RE_BENCH.findall(text):
        if key.endswith("_seconds"):
            out[key] = float(value)
    return out


def wall_seconds(text):
    """Wall-clock aproximado entre el primer y el ultimo timestamp del log."""
    stamps = []
    for stamp, msec in _RE_TS.findall(text):
        try:
            stamps.append(datetime.strptime(f"{stamp},{msec}", "%Y-%m-%d %H:%M:%S,%f"))
        except ValueError:
            continue
    if len(stamps) < 2:
        return None
    span = (max(stamps) - min(stamps)).total_seconds()
    return span if span > 0 else None


def _is_complete(text):
    return all(marker in text for marker in COMPLETE_MARKERS)


def analyse_log(log, size, origin):
    """Lee un log y devuelve la fila; None si el archivo no existe."""
    if not log.exists():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    row = parse_uce_lines(text)
    if "n_uces" not in row:
        return None
    row["incomplete"] = not _is_complete(text)
    row["wall_seconds"] = None if row["incomplete"] else wall_seconds(text)
    row["log"] = str(log)
    row["origin"] = origin
    if "size" in row and row["size"] != size:
        print(
            f"  [!] el log declara uce_target_size={row['size']}; "
            f"se indexa como {row['size']} (valor real), no como {size} (nombre del log)"
        )
    else:
        row["size"] = size
    return row


def discover_logs():
    """(size, path) de todos los data/diag/bench_uce{N}.log, ordenados por N."""
    if not DIAG_DIR.exists():
        return []
    found = []
    for path in sorted(DIAG_DIR.glob("bench_uce*.log")):
        m = _RE_LOG_NAME.search(path.name)
        if m:
            found.append((int(m.group(1)), path))
    return found


def linfit(x, y):
    """Ajuste lineal y = b*x + a. Devuelve (b, a, r2) o None si degenera."""
    n = len(x)
    if n < 2 or len(set(x)) < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx == 0:
        return None
    b = sum((v - mx) * (w - my) for v, w in zip(x, y)) / sxx
    a = my - b * mx
    ss_tot = sum((w - my) ** 2 for w in y)
    ss_res = sum((w - (a + b * v)) ** 2 for v, w in zip(x, y))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else (1.0 if ss_res == 0 else 0.0)
    return b, a, r2


def inverse_x(fit, y):
    b, a, _ = fit
    return (y - a) / b if b else None


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def mode_int(values):
    vals = [v for v in values if v]
    return max(set(vals), key=vals.count) if vals else None


def _fmt(value, width, precision=1):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:{width}.{precision}f}"


def build_models(rows):
    """Ajustes y constantes derivados de TODAS las filas (todos los logs)."""
    sized = [r for r in rows if r.get("n_uces") and r.get("size")]
    models = {"rows": sized}
    models["product"] = mean_or_none([r["size"] * r["n_uces"] for r in sized])
    models["corpus_words"] = mean_or_none(
        [
            r["n_uces"] * r["mean_words_per_uce"]
            for r in sized
            if r.get("mean_words_per_uce") is not None
        ]
    )
    models["n_docs"] = mode_int([r.get("n_docs") for r in sized])
    xs = [r["size"] for r in sized]
    models["fit_n"] = linfit(xs, [r["n_uces"] for r in sized])
    if all(r.get("mean_words_per_uce") is not None for r in sized):
        models["fit_mw"] = linfit(xs, [r["mean_words_per_uce"] for r in sized])
    else:
        models["fit_mw"] = None
    return models


def cost_model(rows):
    """(s/UCE medio, regresion wall_seconds ~ n_UCE, puntos) con filas completas."""
    pts = [
        (r["n_uces"], r["wall_seconds"])
        for r in rows
        if r.get("wall_seconds") and r.get("n_uces")
    ]
    avg = mean_or_none([w / n for n, w in pts])
    fit = linfit([p[0] for p in pts], [p[1] for p in pts])
    return avg, fit, pts


def predict_cost(n_uce, avg_s_uce, fit):
    """(coste por media s/UCE, coste por regresion) para n_uce."""
    by_avg = avg_s_uce * n_uce if avg_s_uce is not None else None
    by_fit = (fit[0] * n_uce + fit[1]) if fit else None
    return by_avg, by_fit


def _in_bounds(target):
    return UCE_BOUNDS[0] <= target <= UCE_BOUNDS[1]


def main():
    _utf8_stdout()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="targets a LANZAR (no limita el ajuste, que usa todos los logs); "
        f"por defecto {list(LAUNCH_DEFAULT)}",
    )
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="no lanza nada: solo lee los data/diag/bench_uce{N}.log existentes",
    )
    ap.add_argument(
        "--force-run",
        action="store_true",
        help="ignora el log y relanza SOLO los --sizes (~2 h por tamano)",
    )
    ap.add_argument(
        "--coref",
        action="store_true",
        help="no-op: el pipeline no instrumenta coref por etapa (ver docstring)",
    )
    ap.add_argument("--runner", choices=["docker", "local"], default=None)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument(
        "--extrapolate",
        type=int,
        default=None,
        help="n de UCE de un corpus mayor; estima coste con s/UCE",
    )
    ap.add_argument(
        "--target",
        type=int,
        default=None,
        help="evalua un target hipotetico SIN lanzar nada (predice n_UCE, "
        "pal_cr/UCE y coste)",
    )
    ap.add_argument("--out", default=str(DIAG_DIR / "uce_size_bench.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.parse_only and a.force_run:
        ap.error("--parse-only y --force-run son incompatibles")
    if a.target is not None and a.force_run:
        ap.error("--target no lanza nada: incompatible con --force-run")
    if a.coref:
        print(
            "[!] --coref es no-op: main_workflow.py no expone tiempos por etapa. "
            "Aislar coref exige instrumentarlo; no se pasa --bench-coref."
        )

    no_launch = a.parse_only or a.target is not None
    mode = (
        "parse-only"
        if a.parse_only
        else ("force-run" if a.force_run else ("target" if no_launch else "auto"))
    )
    sizes_label = a.sizes if a.sizes is not None else list(LAUNCH_DEFAULT)
    print(f"modo={mode}; sizes(lanzar)={sizes_label}; out={a.out}")
    print(
        "[i] el ajuste usa TODOS los data/diag/bench_uce*.log; --sizes solo decide que se lanza"
    )

    rows_by_size = {}

    # 1) Descubrimiento: todo log presente entra en el ajuste, sea o no --sizes.
    print("\n== logs descubiertos (entran en el ajuste) ==")
    discovered = discover_logs()
    if not discovered:
        print("  [i] ninguno")
    for size, log in discovered:
        row = analyse_log(log, size, "log")
        if row is None:
            print(f"  [X] {log} sin la linea '<N> UCEs en <M> documentos...'")
            continue
        rows_by_size[row["size"]] = row
        tag = "[!]" if row["incomplete"] else "[ok]"
        wall = "n/a" if row["wall_seconds"] is None else f"{row['wall_seconds']:.0f}s"
        print(
            f"  {tag} size={row['size']}: {row['n_uces']} UCE en {row['n_docs']} docs; "
            f"{row['mean_words_per_uce']:.1f} pal_cr/UCE; wall={wall}; "
            f"{'PARCIAL' if row['incomplete'] else 'ok'}"
        )

    # 2) Lanzamiento: SOLO --sizes, y solo si no es parse-only/target.
    if no_launch:
        for size in a.sizes or []:
            log = DIAG_DIR / f"bench_uce{size}.log"
            if not log.exists():
                print(f"  [X] falta {log}; modo {mode} no lanza el pipeline")
    else:
        for size in a.sizes if a.sizes is not None else LAUNCH_DEFAULT:
            log = DIAG_DIR / f"bench_uce{size}.log"
            if log.exists() and not a.force_run:
                print(f"\n== size={size} == [i] log existente: se reutiliza ({log})")
                continue
            print(f"\n== size={size} == [i] se lanza el pipeline (~2 h)")
            runner = detect_runner(a.runner)
            run_to_log(
                workflow_cmd(runner, "--force", "--uce-target-size", str(size)),
                log,
                a.timeout,
                a.dry_run,
            )
            if a.dry_run:
                continue
            row = analyse_log(log, size, "ahora")
            if row is None:
                print(f"  [X] {log} sin la linea '<N> UCEs en <M> documentos...'")
                continue
            rows_by_size[row["size"]] = row
            tag = "[!]" if row["incomplete"] else "[ok]"
            wall = (
                "n/a" if row["wall_seconds"] is None else f"{row['wall_seconds']:.0f}s"
            )
            print(
                f"  {tag} size={row['size']}: {row['n_uces']} UCE en {row['n_docs']} docs; "
                f"{row['mean_words_per_uce']:.1f} pal_cr/UCE; wall={wall}; "
                f"{'PARCIAL' if row['incomplete'] else 'ok'}"
            )

    rows = sorted(rows_by_size.values(), key=lambda r: r["size"])
    if not rows:
        print("\n[X] sin filas medidas: nada que escribir.")
        return

    if not a.dry_run:
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)

    extra_keys = [k for k in TIME_KEYS if any(k in r for r in rows)]
    print("\n" + "=" * 78)
    print("[i] pal_cr/UCE = palabras CRUDAS por UCE (lo que mide el log).")
    print("[i] NO es la unidad de ALCESTE: sus 13.95 son palabras ANALIZADAS/UCE.")
    header = (
        f"{'size':>5} {'n_UCE':>6} {'pal_cr/UCE':>11} {'UCE/doc':>8} "
        f"{'wall_s':>9} {'s/UCE':>7}"
    )
    for key in extra_keys:
        header += f" {SHORT_TIME.get(key, key):>9}"
    header += f"  {'origen':>6}  estado"
    print(header)
    for r in rows:
        uce_per_doc = (r["n_uces"] / r["n_docs"]) if r.get("n_docs") else None
        s_per_uce = r["wall_seconds"] / r["n_uces"] if r.get("wall_seconds") else None
        line = (
            f"{r['size']:5d} {r['n_uces']:6d} {r['mean_words_per_uce']:11.1f} "
            f"{_fmt(uce_per_doc, 8)} {_fmt(r.get('wall_seconds'), 9, 0)} "
            f"{_fmt(s_per_uce, 7, 2)}"
        )
        for key in extra_keys:
            line += " " + _fmt(r.get(key), 9)
        line += f"  {r.get('origin', '?'):>6}  {'PARCIAL' if r['incomplete'] else 'ok'}"
        print(line)

    ref_n = ALCESTE_REF["n_uce"]
    print(
        f"\nALCESTE 2012 (mismo corpus): size={ALCESTE_REF['uce_size']} -> {ref_n} UCE "
        f"(26 ocurrencias/UCE; {ALCESTE_REF['words_per_uce_analyzed']} palabras "
        "analizadas/UCE)"
    )
    near = min(rows, key=lambda r: abs(r["n_uces"] - ref_n))
    print(
        f"Fila observada mas cercana: size={near['size']} -> {near['n_uces']} UCE "
        f"(dif={near['n_uces'] - ref_n:+d})"
    )

    partial = [r for r in rows if r["incomplete"]]
    if partial:
        print(
            f"[i] {len(partial)} fila(s) PARCIAL(es) "
            f"({', '.join(str(r['size']) for r in partial)}): cuentan para "
            "n_UCE/pal_cr (la segmentacion va al principio), no para wall_s."
        )

    models = build_models(rows)
    targets = [r["size"] for r in rows]
    lo, hi = min(targets), max(targets)

    if len(rows) < 2:
        print("[X] se necesitan >=2 filas para ajustar.")
        if a.target is not None:
            print("[X] --target necesita el ajuste: no se puede predecir.")
        print(f"\nResultados: {a.out}")
        return

    # --- Ajustes ---
    fit_n = models["fit_n"]
    fit_mw = models["fit_mw"]
    product = models["product"]
    if fit_n:
        slope, intercept, r2 = fit_n
        print(
            f"n_UCE ~ target: pendiente={slope:+.2f} UCE/target, "
            f"ordenada={intercept:+.1f}, R^2={r2:.4f}"
        )
        est = inverse_x(fit_n, ref_n)
        if est is not None:
            zone = "dentro del rango medido" if lo <= est <= hi else "EXTRAPOLACION"
            print(f"  inversa lineal para {ref_n} UCE: target ~ {est:.1f} ({zone})")
        if product:
            print(
                f"  n_UCE*target ~ {product:.0f} (constante: modelo hiperbolico) "
                f"-> target ~ {product / ref_n:.1f} para {ref_n} UCE"
            )
        if est is not None and product and abs(est - product / ref_n) > 2:
            print(
                "  [!] la inversa lineal subestima: n_UCE decae como 1/target, "
                "usa la estimacion hiperbolica."
            )
        if max(est or 0, (product / ref_n) if product else 0) > hi:
            print(
                f"  => el target necesario queda POR ENCIMA del maximo medido ({hi}): "
                "hay que medir tamanos mayores."
            )

    if fit_mw:
        slope, intercept, r2 = fit_mw
        print(
            f"pal_cr/UCE ~ target: pendiente={slope:+.3f}, "
            f"ordenada={intercept:+.2f}, R^2={r2:.4f}"
        )
        total_words = models["corpus_words"]
        if total_words and slope:
            needed = total_words / ref_n
            implied = (needed - intercept) / slope
            print(
                f"  para {ref_n} UCE el corpus (~{total_words:.0f} palabras crudas) "
                f"exigiria ~{needed:.1f} pal_cr/UCE -> target ~ {implied:.1f}"
            )

    # --- Coste normalizado ---
    avg_s_uce, fit_cost, cost_pts = cost_model(rows)
    print("\nCoste normalizado (solo filas completas):")
    if avg_s_uce is not None:
        print(
            f"  s/UCE medio: {avg_s_uce:.2f} s/UCE ({len(cost_pts)} corrida(s): "
            + ", ".join(f"{n}UCE->{w:.0f}s" for n, w in cost_pts)
            + ")"
        )
    else:
        print("  [i] sin wall_s: no hay s/UCE (todas PARCIAL o sin timestamps).")
    if fit_cost:
        slope, intercept, r2 = fit_cost
        print(
            f"  regresion wall_s ~ n_UCE: {slope:.2f} s/UCE, ordenada={intercept:+.0f}s, "
            f"R^2={r2:.3f}"
        )

    if a.extrapolate:
        print(f"\nCoste estimado para un corpus de {a.extrapolate} UCE:")
        by_avg, by_fit = predict_cost(a.extrapolate, avg_s_uce, fit_cost)
        if by_avg is not None:
            print(
                f"  por s/UCE medio: ~{by_avg:.0f}s (~{by_avg / 3600:.1f} h, "
                f"{avg_s_uce:.2f} s/UCE)"
            )
        if by_fit is not None:
            print(f"  por regresion:   ~{by_fit:.0f}s (~{by_fit / 3600:.1f} h)")
        if by_avg is None and by_fit is None:
            print("  [i] n/a: se necesita >=1 corrida completa con wall_s")
        else:
            print(
                "  [!] proyeccion lineal en n_UCE: valida solo si el coste por UCE "
                "no depende del tamano del corpus."
            )

    # --- Target hipotetico ---
    if a.target is not None:
        t = a.target
        print(f"\n== target hipotetico: {t} ==")
        if _in_bounds(t):
            print(
                f"  [ok] dentro del rango de diseno [{UCE_BOUNDS[0]}, {UCE_BOUNDS[1]}]"
            )
        else:
            print(
                f"  [X] FUERA del rango de diseno [{UCE_BOUNDS[0]}, {UCE_BOUNDS[1]}]: "
                "solo alcanzable con --uce-target-size (desactiva la calibracion)"
            )
        if product:
            n_h = product / t
            print(f"  n_UCE predicho (hiperbolico n_UCE*target~cte): ~{n_h:.0f}")
            if fit_n:
                n_l = fit_n[0] * t + fit_n[1]
                print(f"  n_UCE predicho (lineal):                      ~{n_l:.0f}")
            if avg_s_uce is not None:
                cost = avg_s_uce * n_h
                print(
                    f"  coste predicho: ~{cost:.0f}s (~{cost / 3600:.1f} h) "
                    f"a {avg_s_uce:.2f} s/UCE"
                )
            if fit_cost:
                cost = fit_cost[0] * n_h + fit_cost[1]
                print(f"  coste por regresion: ~{cost:.0f}s (~{cost / 3600:.1f} h)")
        else:
            print("  [i] sin n_UCE*target constante: no se puede predecir n_UCE")
        if fit_mw:
            mw_t = fit_mw[0] * t + fit_mw[1]
            print(
                f"  pal_cr/UCE predicho (lineal): ~{mw_t:.1f} (palabras CRUDAS, no analizadas)"
            )
            if models["corpus_words"] and product:
                mw_id = models["corpus_words"] / (product / t)
                print(
                    f"  pal_cr/UCE por identidad (palabras/n_UCE): ~{mw_id:.1f} "
                    "-> coincide por construccion, no es una segunda confirmacion"
                )
        t376 = (
            (product / ref_n)
            if product
            else (inverse_x(fit_n, ref_n) if fit_n else None)
        )
        if t376 is not None:
            flag = "dentro" if _in_bounds(t376) else "FUERA de [14, 30]"
            print(
                f"  target para {ref_n} UCE: ~{t376:.1f} ({flag}); la brecha "
                f"{ref_n} -> n_UCE con target={t} es de COBERTURA, no de tamano de UCE"
            )

    # --- Lectura / reencuadre ---
    ref_occ = ALCESTE_REF["n_uce"] * ALCESTE_REF["uce_size"]  # 9776 ocurrencias
    corpus_words = models["corpus_words"]
    n_docs = models["n_docs"]
    print("\n" + "=" * 78)
    print("Lectura (aritmetica explicita):")
    print(
        "  [unidad] pal_cr/UCE son palabras CRUDAS/UCE (lo que mide el log). ALCESTE "
        "reporta 13.95 palabras ANALIZADAS/UCE tras seleccion: no compares 23.9 crudas "
        "con 13.95 analizadas; solo la aritmetica de ocurrencias es comparable."
    )

    if fit_mw:
        b, c, _ = fit_mw
        est_unit = inverse_x(fit_mw, ALCESTE_REF["uce_size"])
        if est_unit is not None:
            rlo, rhi = round(est_unit) - 2, round(est_unit) + 2
            inside = _in_bounds(est_unit)
            print(
                f"  [target unit-fiel] pal_cr/UCE = {b:.3f}*target + {c:.2f}; igualar "
                f"{ALCESTE_REF['uce_size']} (ocurrencias/UCE de ALCESTE) -> target ~ "
                f"{est_unit:.1f} (rango {rlo}-{rhi}); "
                + (
                    "dentro del rango de diseno."
                    if inside
                    else "FUERA del rango de diseno."
                )
            )
            t376 = (product / ref_n) if product else est_unit
            mw376 = b * t376 + c
            mw_unit = b * est_unit + c
            if mw_unit:
                print(
                    f"  [coste de igualar {ref_n}] target ~ {t376:.1f} -> pal_cr/UCE ~ "
                    f"{mw376:.1f} = +{(mw376 / mw_unit - 1) * 100:.0f}% respecto a "
                    f"{mw_unit:.1f}: alarga la UCE y destruye el objeto de analisis."
                )
    else:
        print("  [i] sin ajuste pal_cr/UCE: no se resuelve el target unit-fiel.")

    print(
        f"  [rango] el rango de diseno es [{UCE_BOUNDS[0]}, {UCE_BOUNDS[1]}] "
        "(uce_calibration_bounds); target=42 queda FUERA de especificacion y solo se "
        "alcanza con --uce-target-size, que desactiva calibrar_uce_target_size()."
    )

    if corpus_words:
        pct = 100.0 * ref_occ / corpus_words
        missing = corpus_words - ref_occ
        per_doc = missing / n_docs if n_docs else None
        print(
            f"  [cobertura] ALCESTE: {ALCESTE_REF['n_uce']} UCE * "
            f"{ALCESTE_REF['uce_size']} = {ref_occ} ocurrencias = {pct:.0f}% de las "
            f"~{corpus_words:.0f} palabras crudas del corpus -> faltan ~{missing:.0f} "
            f"tokens"
            + (f" (~{per_doc:.0f} por documento)." if per_doc is not None else ".")
        )
        at_ref = next(
            (r for r in rows if r["size"] == ALCESTE_REF["uce_size"]),
            min(rows, key=lambda r: abs(r["n_uces"] - ref_n)),
        )
        print(
            f"           La diferencia {at_ref['n_uces']} (size={at_ref['size']}) vs "
            f"{ref_n} es COBERTURA (que texto entra), no tamano de UCE."
        )

    if product:
        print(
            f"  [!] los dos estimadores NO son independientes: n_UCE*target ~ "
            f"{product:.0f} (constante) y pal_cr/UCE lineal son la MISMA identidad "
            "porque el corpus tiene un numero fijo de palabras. Que coincidan no es "
            "confirmacion doble."
        )

    if a.dry_run:
        print(f"\n[i] --dry-run: no se escribio {a.out}")
    else:
        print(f"\nResultados: {a.out}")


if __name__ == "__main__":
    main()
