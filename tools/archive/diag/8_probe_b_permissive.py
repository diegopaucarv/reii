#!/usr/bin/env python3
"""
Script 8 - Sonda B: Method B (Similarity) esta muerto o solo mal parametrizado?

Hecho que motiva la sonda (corrida de diagnostico): en el STABILITY REPORT
  A Word-count (A1<->A2): 163  (30.9%)  ARI=0.139
  B Similarity (B1<->B2):   0  ( 0.0%)  ARI=0.000
  C Embedding  (C1<->C2):  45  ( 8.5%)
  A<->B 0   B<->C 0   A<->C 14   TRIPLE 0
Con B=0 y B<->C=0, el "triple match" es imposible por construccion y el
optimizador de la etapa 2 devuelve 100% centinelas (-2,-1,0). No es un problema
de tuning: es estructural o es acoplamiento de parametros. Esta sonda distingue
las dos cosas con UNA sola corrida.

Que hace:
  1. Respalda data/best_params.json -> data/best_params.json.pre_probe_b
  2. Muta los parametros hacia valores PERMISIVOS (mas fusion, menos ruido):
        swap_iterations             = 5     (maximo: mas intentos de estabilizar)
        min_cluster_size_cdh        = 0.05  (minimo: permite clases pequenas)
        coref_weight                = 0.4   (maximo: mas senal de coref)
        similarity_threshold        = 0.25  (minimo: el mas permisivo del espacio)
     Conserva "corpus_fingerprint" para que el pipeline NO re-optimice (si
     re-optimizara, la sonda mediria otra cosa y tardaria ~1 h mas).
  3. Lanza el pipeline COMPLETO (~2 h) con --force, en streaming (sin silencio).
  4. Verifica en el log "Found existing best_params.json" y la AUSENCIA de
     "Re-optimizando": si re-optimizo, la sonda NO es valida y lo dice.
  5. Extrae el ULTIMO bloque STABILITY REPORT y la linea de UCEs.
  6. Veredicto:
        B sigue en 0  -> Method B esta ESTRUCTURALMENTE muerto (no es tuning):
                         el problema esta en la construccion de UCEs por cohesion
                         o en el vocabulario/HDBSCAN de Method B, no en los
                         hiperparametros. Hay que instrumentar _clasificar_umbral_similarity.
        B > 0         -> B responde a parametros: era ACOPLAMIENTO. Se pasa a
                         sonda de bisturi (una dimension por corrida).
  7. Restaura best_params.json en finally (siempre, incluso con Ctrl-C).

Uso (host, como 2_/4_/5_: necesita diag_common y lanzar docker):
  python 8_probe_b_permissive.py --dry-run
  python 8_probe_b_permissive.py                       # ~2 h
  python 8_probe_b_permissive.py --param coref_weight=0.4 --param swap_iterations=5
  python 8_probe_b_permissive.py --parse-only           # re-analiza data/diag/probe_b.log

Salida ASCII pura: la consola cp1252 no tiene los glifos del bloque STABILITY.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from reii.diag.common import BEST_PARAMS, DIAG_DIR, detect_runner, workflow_cmd

# Valores permisivos: maximizan la probabilidad de que B produzca algun UC estable.
PRESET = {
    "swap_iterations": 5,
    "min_cluster_size_cdh": 0.05,
    "coref_weight": 0.4,
    "similarity_threshold": 0.25,
}

# Referencia medida en la corrida de diagnostico (para calcular deltas).
BASELINE = {
    "total": 527,
    "a": 163,
    "b": 0,
    "c": 45,
    "ab": 0,
    "bc": 0,
    "ac": 14,
    "triple": 0,
}

_NUM = r"(\d+)"
_RE_TOTAL = re.compile(r"Total UCEs:\s*" + _NUM)
_RE_WITHIN = {
    # formato triple/two_of_three: "A  Word-count   (A1<->A2):   163  ( 30.9%)  ARI=0.139"
    "a": re.compile(
        r"A\s+Word-count\s+\(A1.A2\):\s*"
        + _NUM
        + r"\s*\(\s*([\d.]+)%\)\s*ARI=([-\d.]+)"
    ),
    "b": re.compile(
        r"B\s+Similarity\s+\(B1.B2\):\s*"
        + _NUM
        + r"\s*\(\s*([\d.]+)%\)\s*ARI=([-\d.]+)"
    ),
    "c": re.compile(
        r"C\s+Embedding\s+\(C1.C2\):\s*" + _NUM + r"\s*\(\s*([\d.]+)%\)\s*ARI=([-\d.]+)"
    ),
}
_RE_WITHIN_ALT = {
    # formato dual (wc_coref): "WC stable    A1<->A2: 163 (...)  ARI=..."
    "a": re.compile(
        r"WC\s+stable\s+A1.A2:\s*" + _NUM + r"\s*\(\s*([\d.]+)%\)\s*ARI=([-\d.]+)"
    ),
    "b": re.compile(
        r"SIM\s+stable\s+B1.B2:\s*" + _NUM + r"\s*\(\s*([\d.]+)%\)\s*ARI=([-\d.]+)"
    ),
}
_RE_CROSS = {
    "ab": re.compile(r"A.B cross-stable:\s*" + _NUM),
    "bc": re.compile(r"B.C cross-stable:\s*" + _NUM),
    "ac": re.compile(r"A.C cross-stable:\s*" + _NUM),
}
_RE_TRIPLE = re.compile(
    r"(?:TRIPLE INTERSECTION|TWO-OF-THREE INTERSECTION)\s*(?:\(A.B.C\))?:\s*" + _NUM
)
_RE_UCE_SUMMARY = re.compile(
    r"(\d+)\s+UCEs\s+en\s+(\d+)\s+documentos\.\s+Promedio real:\s+([0-9.]+)\s+palabras/UCE"
)


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def parse_stability(text: str) -> dict:
    """Ultimo bloque STABILITY del log -> dict de conteos.

    Los glifos del bloque (box-drawing, flechas) no sobreviven a cp1252, asi
    que el parseo se apoya solo en las etiquetas ASCII y en el orden de los
    numeros, nunca en el dibujo.
    """
    out = {}
    total = _RE_TOTAL.findall(text)
    if total:
        out["total"] = int(total[-1])
    for key, rx in _RE_WITHIN.items():
        hits = rx.findall(text)
        if hits:
            n, pct, ari = hits[-1]
            out[key] = int(n)
            out[f"{key}_pct"] = float(pct)
            out[f"{key}_ari"] = float(ari)
    for key, rx in _RE_WITHIN_ALT.items():
        if key in out:
            continue
        hits = rx.findall(text)
        if hits:
            n, pct, ari = hits[-1]
            out[key] = int(n)
            out[f"{key}_pct"] = float(pct)
            out[f"{key}_ari"] = float(ari)
    for key, rx in _RE_CROSS.items():
        hits = rx.findall(text)
        if hits:
            out[key] = int(hits[-1])
    hits = _RE_TRIPLE.findall(text)
    if hits:
        out["triple"] = int(hits[-1])
    return out


def parse_uce_summary(text: str) -> dict:
    hits = _RE_UCE_SUMMARY.findall(text)
    if not hits:
        return {}
    n_uces, n_docs, mean = hits[-1]
    return {"n_uces": int(n_uces), "n_docs": int(n_docs), "words_per_uce": float(mean)}


def run_streaming(cmd, log_path: Path, timeout: int) -> int | None:
    """Ejecuta cmd haciendo tee a stdout y al log (sin silencio de 2 h)."""
    import threading

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("  $ " + " ".join(cmd))
    expired = {"hit": False}

    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).resolve().parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        def _kill():
            expired["hit"] = True
            proc.kill()

        timer = threading.Timer(timeout, _kill)
        timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
            rc = proc.wait()
        finally:
            timer.cancel()
    if expired["hit"]:
        print(f"  [!] timeout ({timeout}s); log parcial en {log_path}")
        return None
    return rc


def _load_params() -> dict:
    data = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))
    params = data.get("best_params") or data.get("params") or {}
    return {"data": data, "params": dict(params)}


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runner", choices=["docker", "local"], default=None)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="K=V",
        help="override extra (repetible), p.ej. --param tsj=4",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--parse-only", action="store_true", help="re-analiza el log sin lanzar nada"
    )
    ap.add_argument(
        "--keep-mutated",
        action="store_true",
        help="NO restaura best_params.json (solo para inspeccion)",
    )
    ap.add_argument("--log", default=str(DIAG_DIR / "probe_b.log"))
    ap.add_argument("--out", default=str(DIAG_DIR / "probe_b.json"))
    a = ap.parse_args()

    log_path = Path(a.log)
    overrides = dict(PRESET)
    for item in a.param:
        if "=" not in item:
            print(f"[X] --param espera K=V, recibi {item!r}")
            return 2
        key, value = item.split("=", 1)
        try:
            overrides[key.strip()] = (
                int(value) if value.strip().lstrip("-").isdigit() else float(value)
            )
        except ValueError:
            overrides[key.strip()] = value.strip()

    if a.parse_only:
        if not log_path.exists():
            print(f"[X] no existe {log_path}")
            return 1
        text = log_path.read_text(encoding="utf-8", errors="replace")
    else:
        if not BEST_PARAMS.exists():
            print(
                f"[X] no existe {BEST_PARAMS}; sin best_params el pipeline RE-OPTIMIZA (~1 h extra)"
            )
            return 1
        loaded = _load_params()
        data, params = loaded["data"], loaded["params"]
        before = {k: params.get(k) for k in overrides}
        params.update(overrides)
        data["best_params"] = params
        data["params"] = params
        fingerprint = data.get("corpus_fingerprint")

        print("== sonda B: parametros permisivos ==")
        for key in sorted(overrides):
            print(f"  {key:32s} {before.get(key)!s:>8} -> {params.get(key)}")
        print(f"  corpus_fingerprint conservado: {fingerprint}")

        if a.dry_run:
            tmp = log_path.with_suffix(".dryrun.log")
            print(f"[dry-run] no se escribe {BEST_PARAMS.name} ni se lanza nada")
            print("  " + " ".join(workflow_cmd(detect_runner(a.runner), "--force")))
            print(f"  (el log real iria a {log_path}; el dry-run usa {tmp.name})")
            return 0

        backup = BEST_PARAMS.with_suffix(".json.pre_probe_b")
        shutil.copy2(BEST_PARAMS, backup)
        print(f"  [ok] backup: {backup.name}")
        BEST_PARAMS.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        try:
            cmd = workflow_cmd(detect_runner(a.runner), "--force")
            rc = run_streaming(cmd, log_path, a.timeout)
            if rc != 0:
                print(f"[X] el pipeline termino con returncode={rc}; ver {log_path}")
                return 1
            text = log_path.read_text(encoding="utf-8", errors="replace")
        finally:
            if a.keep_mutated:
                print(
                    f"[!] --keep-mutated: {BEST_PARAMS.name} queda mutado (backup en {backup.name})"
                )
            elif backup.exists():
                shutil.move(str(backup), str(BEST_PARAMS))
                print(f"  [ok] {BEST_PARAMS.name} restaurado desde {backup.name}")

    # ── validez de la sonda ───────────────────────────────────────────────
    used_cached = "Found existing best_params.json" in text
    reoptimized = "Re-optimizando" in text
    if reoptimized:
        print(
            "[!] el log dice 'Re-optimizando': best_params.json NO se aplico "
            "(fingerprint distinto o no se encontro). La sonda mide otra cosa."
        )
    elif used_cached:
        print(
            "[ok] el log confirma que se aplico best_params.json (no hubo re-optimizacion)"
        )
    else:
        print(
            "[!] no encuentro la marca 'Found existing best_params.json' en el log: revisalo a mano"
        )

    stab = parse_stability(text)
    uce = parse_uce_summary(text)
    if not stab:
        print("[X] no hay bloque STABILITY REPORT en el log")
        return 1

    print("\n" + "=" * 72)
    print("RESULTADO DE LA SONDA B (parametros permisivos)")
    print("=" * 72)
    if uce:
        print(
            f"  UCEs: {uce['n_uces']} en {uce['n_docs']} docs "
            f"({uce['words_per_uce']:.1f} palabras/UCE)"
        )
    header = f"{'metodo':<22} {'ahora':>8} {'base':>6} {'delta':>7} {'%':>7} {'ARI':>7}"
    print(header)
    print("-" * len(header))
    for key, label in (
        ("a", "A Word-count"),
        ("b", "B Similarity"),
        ("c", "C Embedding"),
    ):
        now = stab.get(key)
        base = BASELINE.get(key)
        delta = "-" if now is None else f"{now - base:+d}"
        print(
            f"{label:<22} {('n/a' if now is None else now):>8} {base:>6} {delta:>7} "
            f"{stab.get(f'{key}_pct', float('nan')):>7.1f} {stab.get(f'{key}_ari', float('nan')):>7.3f}"
        )
    for key, label in (
        ("ab", "A<->B"),
        ("bc", "B<->C"),
        ("ac", "A<->C"),
        ("triple", "TRIPLE"),
    ):
        now = stab.get(key)
        base = BASELINE.get(key)
        delta = "-" if now is None else f"{now - base:+d}"
        print(f"{label:<22} {('n/a' if now is None else now):>8} {base:>6} {delta:>7}")

    b = stab.get("b", 0)
    bc = stab.get("bc", 0)
    print("\nVEREDICTO:")
    if b == 0:
        print(
            "  B sigue en 0 incluso con los parametros MAS permisivos del espacio.\n"
            "  => Method B esta ESTRUCTURALMENTE muerto, no mal parametrizado.\n"
            "     El objetivo del optimizador (triple interseccion = A n B n C) es\n"
            "     inalcanzable: mientras B=0 todo trial de la etapa 2 sera centinela.\n"
            "     Siguiente paso: instrumentar _clasificar_umbral_similarity (contar\n"
            "     UCs producidos, vocab, ruido HDBSCAN, merge_rate) y NO gastar mas\n"
            "     trials en la etapa 2."
        )
    elif bc == 0:
        print(
            f"  B revivio ({b} estables) pero B<->C sigue en 0: el acoplamiento esta\n"
            "  en el emparejamiento B/C (Hungarian sobre espacios distintos), no en B.\n"
            "  Siguiente paso: sonda de bisturi sobre min_cluster_size_cdh y\n"
            "  similarity_threshold, una dimension por corrida."
        )
    else:
        print(
            f"  B={b} y B<->C={bc}: era ACOPLAMIENTO de parametros. La etapa 2 puede\n"
            "  producir trials validos. Siguiente paso: sonda de bisturi (1 dimension\n"
            "  por corrida) y solo despues re-lanzar la optimizacion."
        )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "overrides": overrides,
                "used_cached_best_params": used_cached,
                "reoptimized": reoptimized,
                "uce": uce,
                "stability": stab,
                "baseline": BASELINE,
                "log": str(log_path),
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResultados: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
