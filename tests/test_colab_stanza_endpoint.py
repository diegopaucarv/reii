#!/usr/bin/env python3
"""
test_colab_stanza_endpoint.py
==============================

Suite de validación end-to-end para el servidor Stanza remoto (Colab +
Cloudflare Tunnel) usado por `reii` vía `/v1/extract_roots` y
`/v1/extract_coref`.

Diseñado para correr DESDE UNA MÁQUINA CON ACCESO DE RED al túnel
(``*.trycloudflare.com``) — típicamente tu laptop/servidor de desarrollo,
NO un sandbox aislado. Usa solo `requests` (ya es dependencia de `reii`).

Qué valida, y por qué (mapeado a los 3 problemas diagnosticados):

  1. Conectividad básica       -> el túnel está vivo y resuelve DNS.
  2. /health bajo carga         -> handlers `def` (no `async def`) + lock de
                                    GPU: /health debe responder incluso con
                                    la GPU ocupada procesando otra request.
                                    Si esto fallara, Cloudflare corta el
                                    túnel por keepalive y el cliente ve
                                    exactamente los errores de
                                    NameResolutionError del log que motivó
                                    este fix.
  3. Concurrencia (N en paralelo) -> ninguna request debe colgarse ni
                                    devolver 5xx por corrida simultánea
                                    sobre la misma GPU.
  4. Auth                       -> token inválido/ausente -> 401, nunca 500.
  5. Idioma no soportado        -> lang != "es" -> 400, nunca 500.
  6. Corrección /v1/extract_roots -> sujetos sintácticos razonables sobre
                                    oraciones españolas de control.
  7. Corrección /v1/extract_coref -> cadenas de correferencia no vacías
                                    sobre un texto con un antecedente claro,
                                    y offsets (start_char/end_char) que
                                    efectivamente apuntan al texto de la
                                    mención dentro del string enviado.
  8. Ventana grande (~1000 chars) -> replica el tamaño máximo de sub-ventana
                                    que usa `_extract_global_chains` en
                                    producción (MAX_WINDOW_CHARS=1000) y mide
                                    si entra bajo el timeout de 90s del
                                    cliente / ~100s del túnel.
  9. Resultado por lote coherente -> len(results) == len(texts) en ambos
                                    endpoints (el cliente real revienta si
                                    no coincide, ver segmentador.py).

Uso:
    python test_colab_stanza_endpoint.py --url https://TU-TUNEL.trycloudflare.com
    python test_colab_stanza_endpoint.py --url ... --token 6b92bb50d646a9a4030ac1adf0a5cc8e
    python test_colab_stanza_endpoint.py --url ... --skip-concurrency --skip-large-window
    python test_colab_stanza_endpoint.py --url ... --report-json report.json

Salida: resumen en consola (PASS/FAIL/SKIP por caso) + reporte JSON opcional
con timings, para pegar en el reporte de validación.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import requests
except ImportError:
    print("Falta 'requests'. Instala con: pip install requests", file=sys.stderr)
    sys.exit(1)

# ── Defaults alineados con src/reii/config.py (STANZA_SERVER_*) ─────────────
DEFAULT_TOKEN = "6b92bb50d646a9a4030ac1adf0a5cc8e"  # token por defecto del colab/stanza_server.py
DEFAULT_TIMEOUT = 90.0  # STANZA_SERVER_TIMEOUT
MAX_WINDOW_CHARS = 1000  # tope de sub-ventana usado por segmentador.py


@dataclass
class CaseResult:
    name: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    detail: str = ""
    elapsed_s: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    base_url: str
    cases: List[CaseResult] = field(default_factory=list)

    def add(self, result: CaseResult) -> None:
        self.cases.append(result)
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️ "}[result.status]
        timing = f" ({result.elapsed_s:.2f}s)" if result.elapsed_s is not None else ""
        print(f"{icon} {result.name}{timing} — {result.detail}")

    def summary(self) -> Dict[str, int]:
        out = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        for c in self.cases:
            out[c.status] += 1
        return out

    def to_json(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "summary": self.summary(),
            "cases": [asdict(c) for c in self.cases],
        }


def _run(report: Report, name: str, fn: Callable[[], CaseResult]) -> None:
    try:
        result = fn()
    except Exception as e:  # noqa: BLE001 — queremos capturar TODO aquí
        result = CaseResult(name=name, status="FAIL", detail=f"excepción no controlada: {e!r}")
    report.add(result)


# ─────────────────────────────────────────────────────────────────────────
# Casos
# ─────────────────────────────────────────────────────────────────────────
def case_dns_and_health(base_url: str, timeout: float) -> CaseResult:
    t0 = time.monotonic()
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        return CaseResult(
            "01 · Conectividad + /health", "FAIL",
            f"no se pudo conectar ({e.__class__.__name__}: {e})",
        )
    except requests.exceptions.Timeout:
        return CaseResult("01 · Conectividad + /health", "FAIL", f"timeout > {timeout}s")
    elapsed = time.monotonic() - t0
    if r.status_code != 200:
        return CaseResult(
            "01 · Conectividad + /health", "FAIL",
            f"status={r.status_code}, body={r.text[:200]!r}", elapsed,
        )
    try:
        body = r.json()
    except ValueError:
        return CaseResult("01 · Conectividad + /health", "FAIL", f"respuesta no-JSON: {r.text[:200]!r}", elapsed)
    if not body.get("model_loaded"):
        return CaseResult("01 · Conectividad + /health", "FAIL", f"model_loaded=False — {body}", elapsed)
    return CaseResult("01 · Conectividad + /health", "PASS", f"{body}", elapsed)


def case_auth_rejected(base_url: str, timeout: float) -> CaseResult:
    t0 = time.monotonic()
    r = requests.post(
        f"{base_url}/v1/extract_roots",
        json={"texts": ["Texto de prueba."], "lang": "es"},
        headers={"Authorization": "Bearer token-invalido-a-proposito"},
        timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    if r.status_code == 401:
        return CaseResult("04 · Token inválido → 401", "PASS", f"status={r.status_code}", elapsed)
    return CaseResult("04 · Token inválido → 401", "FAIL", f"esperado 401, llegó {r.status_code}: {r.text[:200]!r}", elapsed)


def case_bad_lang(base_url: str, token: str, timeout: float) -> CaseResult:
    t0 = time.monotonic()
    r = requests.post(
        f"{base_url}/v1/extract_roots",
        json={"texts": ["Some english text."], "lang": "en"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    if r.status_code == 400:
        return CaseResult("05 · lang≠es → 400", "PASS", f"status={r.status_code}", elapsed)
    return CaseResult("05 · lang≠es → 400", "FAIL", f"esperado 400, llegó {r.status_code}: {r.text[:200]!r}", elapsed)


ROOT_SENTENCES = [
    "El investigador analizó los datos del corpus.",
    "Las enfermeras de primera línea atendieron a los pacientes durante el desastre.",
    "La comunidad de Puente Piedra organizó una asamblea vecinal.",
]


def case_extract_roots(base_url: str, token: str, timeout: float) -> CaseResult:
    t0 = time.monotonic()
    r = requests.post(
        f"{base_url}/v1/extract_roots",
        json={"texts": ROOT_SENTENCES, "lang": "es"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    if r.status_code != 200:
        return CaseResult("06 · /v1/extract_roots correctitud", "FAIL", f"status={r.status_code}: {r.text[:300]!r}", elapsed)
    data = r.json()
    results = data.get("results", [])
    if len(results) != len(ROOT_SENTENCES):
        return CaseResult(
            "06 · /v1/extract_roots correctitud", "FAIL",
            f"len(results)={len(results)} != len(texts)={len(ROOT_SENTENCES)} "
            "(el cliente real de reii revienta con un ValueError aquí)", elapsed,
        )
    empty = [res["text"] for res in results if not res.get("subjects")]
    if empty:
        return CaseResult(
            "06 · /v1/extract_roots correctitud", "FAIL",
            f"{len(empty)}/{len(results)} oraciones sin ningún sujeto extraído: {empty}", elapsed,
            extra={"results": results},
        )
    return CaseResult(
        "06 · /v1/extract_roots correctitud", "PASS",
        f"{len(results)} oraciones, sujetos no vacíos en todas", elapsed,
        extra={"results": results},
    )


COREF_TEXT = (
    "María llegó temprano al centro de salud. Ella revisó el historial de los pacientes "
    "antes de que el doctor Luis comenzara su turno. Él le agradeció el trabajo."
)


def case_extract_coref(base_url: str, token: str, timeout: float) -> CaseResult:
    t0 = time.monotonic()
    r = requests.post(
        f"{base_url}/v1/extract_coref",
        json={"texts": [COREF_TEXT], "lang": "es"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    if r.status_code != 200:
        return CaseResult("07 · /v1/extract_coref correctitud", "FAIL", f"status={r.status_code}: {r.text[:300]!r}", elapsed)
    data = r.json()
    results = data.get("results", [])
    if len(results) != 1:
        return CaseResult("07 · /v1/extract_coref correctitud", "FAIL", f"len(results)={len(results)} != 1", elapsed)
    chains = results[0].get("chains", [])
    if not chains:
        return CaseResult(
            "07 · /v1/extract_coref correctitud", "FAIL",
            "0 cadenas devueltas para un texto con antecedentes explícitos "
            "(María→Ella, Luis→Él) — revisar carga del procesador 'coref'.",
            elapsed,
        )
    # Validar que los offsets apuntan de verdad al texto de la mención.
    offset_errors = []
    for chain in chains:
        for m in chain.get("mentions", []):
            s, e, txt = m.get("start_char"), m.get("end_char"), m.get("text")
            if s is None or e is None:
                offset_errors.append(f"mención sin offsets: {m}")
                continue
            if COREF_TEXT[s:e] != txt:
                offset_errors.append(
                    f"offset desalineado: texto[{s}:{e}]={COREF_TEXT[s:e]!r} != {txt!r}"
                )
    if offset_errors:
        return CaseResult(
            "07 · /v1/extract_coref correctitud", "FAIL",
            f"{len(offset_errors)} error(es) de offset: {offset_errors[:3]}", elapsed,
            extra={"chains": chains},
        )
    return CaseResult(
        "07 · /v1/extract_coref correctitud", "PASS",
        f"{len(chains)} cadena(s), offsets consistentes", elapsed,
        extra={"chains": chains},
    )


def case_large_window(base_url: str, token: str, timeout: float) -> CaseResult:
    # Replica el tamaño máximo real de sub-ventana (segmentador.py:
    # MAX_WINDOW_CHARS = 1000) para medir si entra bajo el timeout de 90s
    # del cliente / ~100s del túnel de Cloudflare.
    base_sentence = (
        "El equipo de Ergalia coordinó con AYNE-Perú la implementación del diagnóstico "
        "comunitario en Puente Piedra, y ella explicó que los resultados se compartirían "
        "con los dirigentes vecinales la próxima semana. "
    )
    text = (base_sentence * (MAX_WINDOW_CHARS // len(base_sentence) + 1))[:MAX_WINDOW_CHARS]
    t0 = time.monotonic()
    try:
        r = requests.post(
            f"{base_url}/v1/extract_coref",
            json={"texts": [text], "lang": "es"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        elapsed = time.monotonic() - t0
        return CaseResult(
            "08 · Ventana de ~1000 chars bajo timeout", "FAIL",
            f"timeout tras {elapsed:.1f}s (límite cliente={timeout}s) — "
            "en producción esto dispara fallback a Stanza local o cadenas vacías.",
            elapsed,
        )
    elapsed = time.monotonic() - t0
    if r.status_code != 200:
        return CaseResult("08 · Ventana de ~1000 chars bajo timeout", "FAIL", f"status={r.status_code}: {r.text[:200]!r}", elapsed)
    margin = timeout - elapsed
    status = "PASS" if margin > 15 else "FAIL"  # <15s de margen es frágil ante jitter del túnel
    return CaseResult(
        "08 · Ventana de ~1000 chars bajo timeout", status,
        f"completó en {elapsed:.1f}s (margen de {margin:.1f}s antes de timeout={timeout}s)", elapsed,
    )


def case_health_under_gpu_load(base_url: str, token: str, timeout: float) -> CaseResult:
    """
    EL CASO CENTRAL del fix. Dispara una request de coref pesada (ocupa la
    GPU varios segundos) y, EN PARALELO, golpea /health repetidamente. Si el
    servidor sigue usando `async def` sin threadpool (o no tiene el lock de
    GPU correctamente liberando el event loop), /health se cuelga mientras
    la GPU trabaja y Cloudflare termina cortando el túnel — exactamente el
    síntoma del log original (NameResolutionError tras ~minutos).
    """
    heavy_text = (COREF_TEXT * 6)[:MAX_WINDOW_CHARS]

    health_results: List[Dict[str, Any]] = []
    stop = {"flag": False}

    def _poll_health():
        while not stop["flag"]:
            t0 = time.monotonic()
            try:
                r = requests.get(f"{base_url}/health", timeout=10)
                health_results.append({"ok": r.status_code == 200, "elapsed": time.monotonic() - t0})
            except Exception as e:
                health_results.append({"ok": False, "elapsed": time.monotonic() - t0, "error": repr(e)})
            time.sleep(1)

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        heavy_future = ex.submit(
            requests.post,
            f"{base_url}/v1/extract_coref",
            json={"texts": [heavy_text], "lang": "es"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        poll_future = ex.submit(_poll_health)
        try:
            heavy_future.result(timeout=timeout + 5)
        except Exception as e:
            stop["flag"] = True
            poll_future.result(timeout=5)
            return CaseResult(
                "02 · /health bajo carga de GPU", "FAIL",
                f"la request pesada de coref falló/colgó: {e!r} "
                f"({len(health_results)} polls de /health capturados)",
            )
        stop["flag"] = True
        poll_future.result(timeout=5)

    if not health_results:
        return CaseResult("02 · /health bajo carga de GPU", "FAIL", "no se capturó ningún poll de /health (ventana demasiado corta)")
    failed_polls = [h for h in health_results if not h["ok"]]
    if failed_polls:
        return CaseResult(
            "02 · /health bajo carga de GPU", "FAIL",
            f"{len(failed_polls)}/{len(health_results)} polls de /health fallaron mientras la GPU "
            f"procesaba: {failed_polls[:3]}",
            extra={"health_results": health_results},
        )
    avg = sum(h["elapsed"] for h in health_results) / len(health_results)
    return CaseResult(
        "02 · /health bajo carga de GPU", "PASS",
        f"{len(health_results)}/{len(health_results)} polls OK mientras la GPU trabajaba "
        f"(latencia media /health={avg:.2f}s) — el event loop no se bloqueó", None,
        extra={"health_results": health_results},
    )


def case_concurrency(base_url: str, token: str, timeout: float, n: int) -> CaseResult:
    """
    N requests simultáneas a /v1/extract_roots. Todas deben devolver 200.
    Ninguna debe colgarse indefinidamente (el lock de GPU las serializa
    server-side, pero el cliente no debe ver timeouts ni 5xx).
    """
    def _hit(i: int) -> Dict[str, Any]:
        t0 = time.monotonic()
        try:
            r = requests.post(
                f"{base_url}/v1/extract_roots",
                json={"texts": [f"Texto de prueba número {i}."], "lang": "es"},
                headers={"Authorization": f"Bearer {token}"},
                timeout=timeout,
            )
            return {"i": i, "status": r.status_code, "elapsed": time.monotonic() - t0}
        except Exception as e:
            return {"i": i, "status": None, "error": repr(e), "elapsed": time.monotonic() - t0}

    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        outcomes = list(ex.map(_hit, range(n)))
    elapsed = time.monotonic() - t0

    bad = [o for o in outcomes if o.get("status") != 200]
    if bad:
        return CaseResult(
            "03 · Concurrencia ({n} en paralelo)".format(n=n), "FAIL",
            f"{len(bad)}/{n} fallaron: {bad[:5]}", elapsed,
            extra={"outcomes": outcomes},
        )
    return CaseResult(
        "03 · Concurrencia ({n} en paralelo)".format(n=n), "PASS",
        f"{n}/{n} con status 200", elapsed,
        extra={"outcomes": outcomes},
    )


def case_batch_consistency(base_url: str, token: str, timeout: float) -> CaseResult:
    texts = [f"Oración número {i} para el lote." for i in range(20)]
    t0 = time.monotonic()
    r = requests.post(
        f"{base_url}/v1/extract_roots",
        json={"texts": texts, "lang": "es"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    if r.status_code != 200:
        return CaseResult("09 · Consistencia de lote (chunking=15)", "FAIL", f"status={r.status_code}", elapsed)
    results = r.json().get("results", [])
    if len(results) != len(texts):
        return CaseResult(
            "09 · Consistencia de lote (chunking=15)", "FAIL",
            f"len(results)={len(results)} != len(texts)={len(texts)} "
            "(lote de 20 cruza el CHUNK_SIZE=15 del servidor — revisar re-ensamblado)", elapsed,
        )
    return CaseResult("09 · Consistencia de lote (chunking=15)", "PASS", f"{len(results)}/{len(texts)} resultados", elapsed)


# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="Base URL del túnel, ej. https://xxx.trycloudflare.com")
    ap.add_argument("--token", default=DEFAULT_TOKEN, help="Bearer token (default: token de ejemplo de colab/stanza_server.py)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Timeout por request en segundos (default: 90, igual que STANZA_SERVER_TIMEOUT)")
    ap.add_argument("--concurrency", type=int, default=20, help="N de requests paralelas para el caso 03 (default: 20)")
    ap.add_argument("--skip-concurrency", action="store_true")
    ap.add_argument("--skip-gpu-load", action="store_true", help="Salta el caso 02 (tarda ~15-30s)")
    ap.add_argument("--skip-large-window", action="store_true")
    ap.add_argument("--report-json", default=None, help="Ruta donde guardar el reporte en JSON")
    args = ap.parse_args()

    base_url = args.url.rstrip("/")
    report = Report(base_url=base_url)

    print(f"Validando servidor Stanza remoto en: {base_url}\n")

    _run(report, "01", lambda: case_dns_and_health(base_url, args.timeout))

    # Si ni siquiera /health responde, no tiene sentido seguir.
    if report.cases[-1].status == "FAIL":
        print("\n⚠️  /health no respondió — se omiten el resto de los casos.")
        print("    Verifica: (a) la celda de Colab sigue corriendo, (b) no pasaron")
        print("    las 6h de límite de sesión, (c) la URL del túnel es la vigente")
        print("    (Cloudflare genera una URL nueva cada vez que se reinicia la celda).")
    else:
        _run(report, "04", lambda: case_auth_rejected(base_url, args.timeout))
        _run(report, "05", lambda: case_bad_lang(base_url, args.token, args.timeout))
        _run(report, "06", lambda: case_extract_roots(base_url, args.token, args.timeout))
        _run(report, "07", lambda: case_extract_coref(base_url, args.token, args.timeout))
        _run(report, "09", lambda: case_batch_consistency(base_url, args.token, args.timeout))

        if args.skip_large_window:
            report.add(CaseResult("08 · Ventana de ~1000 chars bajo timeout", "SKIP", "--skip-large-window"))
        else:
            _run(report, "08", lambda: case_large_window(base_url, args.token, args.timeout))

        if args.skip_gpu_load:
            report.add(CaseResult("02 · /health bajo carga de GPU", "SKIP", "--skip-gpu-load"))
        else:
            _run(report, "02", lambda: case_health_under_gpu_load(base_url, args.token, args.timeout))

        if args.skip_concurrency:
            report.add(CaseResult(f"03 · Concurrencia ({args.concurrency} en paralelo)", "SKIP", "--skip-concurrency"))
        else:
            _run(report, "03", lambda: case_concurrency(base_url, args.token, args.timeout, args.concurrency))

    summary = report.summary()
    total = sum(summary.values())
    print(f"\n{'=' * 60}")
    print(f"RESUMEN: {summary['PASS']} PASS · {summary['FAIL']} FAIL · {summary['SKIP']} SKIP  (de {total})")
    print(f"{'=' * 60}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report.to_json(), f, indent=2, ensure_ascii=False)
        print(f"\nReporte JSON guardado en: {args.report_json}")

    return 1 if summary["FAIL"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
