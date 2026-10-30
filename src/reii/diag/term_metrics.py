"""``term-metrics`` — métricas de términos comparables con ALCESTE, y F9.

Cierra tres de las acciones abiertas del borrador de la sección comparativa
(``docs/design_v5/SECCION_ALCESTE_BORRADOR.md``, §D):

  (b) **phi sin corrección de Yates.** El pipeline guarda ``phi`` con Yates, pero
      ALCESTE no lo aplica, de modo que ambos phi no son magnitudes comparables.
      Aquí se recalcula phi sin Yates desde los recuentos 2x2 que ya están en el
      artefacto (frecuencia_global, frecuencia_cluster, tamaño de clase, n), sin
      relanzar el pipeline.
  (c) **flag de significacion con FDR.** El flag histórico ``significativo`` usa
      el p-valor crudo. Aquí se calcula `significativo_fdr` (p_adj < alpha, el
      Benjamini-Hochberg que el pipeline ya calcula por clase) y se cuantifica
      cuantos terminos cambian de estado.
  (d) **plano factorial F9 reproducible.** `--figure` dibuja el biplot con el
      nivel de PRESENCIA de cada termino (nivel __1, no __0) y con la inercia
      explicada por eje sobre la inercia TOTAL del MCA, no sobre la suma de los
      dos autovalores retenidos.

Verificación incluida: reconstruye la tabla 2x2 desde los recuentos y comprueba
que el chi-cuadrado de Yates y el phi resultantes coinciden con los persistidos.
Si no cuadran, el artefacto no es el que este comando supone y sale con rc=1.

No muta nada: escribe un JSON aparte (``data/diag/paper_terms.json``) y, si se
pide, una figura.

Uso:
  python diag.py term-metrics
  python diag.py term-metrics --top 8
  python diag.py term-metrics --figure docs/figs/F9_plano_factorial.svg
"""

from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from reii.diag.common import REPO, utf8_stdout

DEFAULT_ARTIFACT = REPO / "data" / "workflow_data.json"
DEFAULT_OUT = REPO / "data" / "diag" / "paper_terms.json"

# El artefacto redondea a 4 decimales; 1e-2 deja margen y sigue detectando un
# artefacto que no corresponde a estos supuestos.
VERIFY_TOL = 1e-2


# ── estadística de la tabla 2×2 ────────────────────────────────────────────


def chi2_yates(a: int, b: int, c: int, d: int) -> float:
    """Chi-cuadrado con corrección de continuidad de Yates."""
    n = a + b + c + d
    if n <= 0 or min(a + b, c + d, a + c, b + d) <= 0:
        return 0.0
    num = abs(a * d - b * c) - n / 2.0
    if num <= 0:
        return 0.0
    return num**2 * n / ((a + b) * (c + d) * (a + c) * (b + d))


def phi_signed(a: int, b: int, c: int, d: int) -> float:
    """phi con signo y SIN corrección de continuidad (la familia de ALCESTE)."""
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den <= 0:
        return 0.0
    return (a * d - b * c) / math.sqrt(den)


def phi_yates_signed(a: int, b: int, c: int, d: int) -> float:
    """phi con signo y con Yates: lo que el pipeline persiste en `phi`."""
    n = a + b + c + d
    if n <= 0:
        return 0.0
    chi2 = chi2_yates(a, b, c, d)
    if chi2 <= 0:
        return 0.0
    signo = 1.0 if a * d - b * c >= 0 else -1.0
    return signo * math.sqrt(chi2 / n)


def class_sizes(labels) -> dict:
    """{cluster: n} contando solo las UCE con clase asignada (>= 0)."""
    out = {}
    for label in labels:
        k = int(label)
        if k < 0:
            continue
        out[k] = out.get(k, 0) + 1
    return out


# ── derivación sobre el artefacto ──────────────────────────────────────────


def _contingency(term, sizes, n_ucs):
    """(a, b, c, d) del término dentro de su clase, o None si no cuadra.

    a = UCE de la clase con el término      b = UCE de fuera con el término
    c = UCE de la clase sin el término      d = UCE de fuera sin el término
    """
    cluster = int(term["cluster"])
    n_class = sizes.get(cluster)
    if n_class is None:
        return None
    a = int(term["frecuencia_cluster"])
    n_global = int(term.get("frecuencia_global") or 0)
    c = n_class - a
    b = n_global - a
    d = n_ucs - n_class - b
    if min(a, b, c, d) < 0:
        return None
    return a, b, c, d


def derive_terms(artifact: dict, fdr_alpha: float):
    """Filas enriquecidas (phi sin Yates, flag FDR) + metadatos de validación."""
    mv = artifact.get("multivariate") or {}
    sizes = class_sizes(mv.get("cluster_labels") or [])
    n_ucs = sum(sizes.values())
    terms = artifact.get("terminos") or []

    rows = []
    skipped = 0
    max_dchi = 0.0
    max_dphi = 0.0
    for term in terms:
        cells = _contingency(term, sizes, n_ucs)
        if cells is None:
            skipped += 1
            continue
        a, b, c, d = cells
        chi2_val = chi2_yates(a, b, c, d)
        phi_yates_val = phi_yates_signed(a, b, c, d)
        phi_raw_val = phi_signed(a, b, c, d)

        stored_chi2 = term.get("chi2_yates")
        stored_phi = term.get("phi")
        if stored_chi2 is not None:
            max_dchi = max(max_dchi, abs(float(stored_chi2) - chi2_val))
        if stored_phi is not None:
            max_dphi = max(max_dphi, abs(float(stored_phi) - phi_yates_val))

        p_adj = term.get("p_adj")
        rows.append(
            {
                "termino": term.get("termino"),
                "cluster": int(term["cluster"]),
                "frecuencia_global": int(term.get("frecuencia_global") or 0),
                "frecuencia_cluster": a,
                "n_clase": sizes[int(term["cluster"])],
                "chi2_yates": round(chi2_val, 6),
                "phi": round(phi_yates_val, 6),
                "phi_sin_yates": round(phi_raw_val, 6),
                "p_valor": term.get("p_valor"),
                "p_adj": p_adj,
                "significativo": bool(term.get("significativo")),
                "significativo_fdr": bool(
                    p_adj is not None and float(p_adj) < fdr_alpha
                ),
            }
        )

    meta = {
        "n_uces_con_clase": n_ucs,
        "class_sizes": dict(sorted(sizes.items())),
        "n_term_rows": len(terms),
        "n_rows_valid": len(rows),
        "n_rows_skipped": skipped,
        "max_abs_diff_chi2_yates": round(max_dchi, 6),
        "max_abs_diff_phi": round(max_dphi, 6),
    }
    return rows, meta


def mca_meta(artifact: dict) -> dict:
    """Inercia por eje corregida (sobre la inercia total) y referencia del total."""
    mv = artifact.get("multivariate") or {}
    eigen = [float(v) for v in (mv.get("eigenvalues") or [])]
    stored = [float(v) for v in (mv.get("explained_inertia") or [])]
    raw_cols = len(mv.get("col_coords") or [])
    n_terms = len(mv.get("terms") or [])
    total = mv.get("total_inertia")
    if total is None:
        # MCA sobre Q variables binarias: J columnas indicadoras = 2Q, y la
        # inercia total vale J/Q - 1 = 1. Se deduce de la duplicación.
        total = (raw_cols / n_terms - 1.0) if n_terms and raw_cols > n_terms else 1.0
    total = float(total) or 1.0
    corrected = [e / total for e in eigen]
    return {
        "eigenvalues": eigen,
        "total_inertia": total,
        "explained_inertia_corrected": corrected,
        "explained_inertia_stored": stored,
    }


def summarise(rows) -> dict:
    per_class = {}
    for row in rows:
        acc = per_class.setdefault(
            row["cluster"], {"n_terms": 0, "sig_raw": 0, "sig_fdr": 0}
        )
        acc["n_terms"] += 1
        acc["sig_raw"] += int(row["significativo"])
        acc["sig_fdr"] += int(row["significativo_fdr"])
    return {str(k): v for k, v in sorted(per_class.items())}


def yates_deflation(rows) -> dict:
    ratios = [
        row["phi"] / row["phi_sin_yates"]
        for row in rows
        if abs(row["phi_sin_yates"]) > 1e-9 and abs(row["phi"]) > 1e-9
    ]
    if not ratios:
        return {}
    ratios.sort()
    mid = len(ratios) // 2
    median = ratios[mid] if len(ratios) % 2 else (ratios[mid - 1] + ratios[mid]) / 2
    return {"median": median, "min": ratios[0], "max": ratios[-1], "n": len(ratios)}


# ── informe ────────────────────────────────────────────────────────────────


def print_report(artifact_path, meta, info, per_class, deflation, rows, top):
    print("== term-metrics: phi sin Yates y flag FDR ==")
    print(f"artefacto: {artifact_path}")
    sizes = " / ".join(str(v) for _, v in sorted(meta["class_sizes"].items()))
    print(
        f"UCE con clase: {meta['n_uces_con_clase']} ({sizes}) · "
        f"filas de términos: {meta['n_rows_valid']}/{meta['n_term_rows']}"
    )
    if meta["n_rows_skipped"]:
        print(f"[!] filas descartadas (celdas 2x2 inválidas): {meta['n_rows_skipped']}")

    worst = max(meta["max_abs_diff_chi2_yates"], meta["max_abs_diff_phi"])
    print(
        "verificación 2x2 vs artefacto: Δchi2="
        f"{meta['max_abs_diff_chi2_yates']:.3g} · Δphi="
        f"{meta['max_abs_diff_phi']:.3g} → "
        + ("OK" if worst <= VERIFY_TOL else "NO CUADRA")
    )

    if info["eigenvalues"]:
        pct = " / ".join(
            f"{v * 100:.2f} %" for v in info["explained_inertia_corrected"]
        )
        print(f"MCA: inercia por eje = {pct} (total {info['total_inertia']:.3f})")
        if info["explained_inertia_stored"]:
            old = " / ".join(
                f"{v * 100:.2f} %" for v in info["explained_inertia_stored"]
            )
            print(
                f"     el artefacto guarda {old}: es la parte relativa ENTRE los "
                "dos ejes retenidos, no sobre la inercia total"
            )

    header = f"{'clúster':>8} {'UCE':>6} {'sig p crudo':>12} {'sig FDR':>8} {'Δ':>6}"
    print("\n" + header)
    print("-" * len(header))
    for cluster, acc in per_class.items():
        n_uce = meta["class_sizes"].get(int(cluster), 0)
        print(
            f"{cluster:>8} {n_uce:>6} {acc['sig_raw']:>12} "
            f"{acc['sig_fdr']:>8} {acc['sig_fdr'] - acc['sig_raw']:>+6}"
        )
    total_raw = sum(v["sig_raw"] for v in per_class.values())
    total_fdr = sum(v["sig_fdr"] for v in per_class.values())
    print(
        f"{'total':>8} {meta['n_uces_con_clase']:>6} {total_raw:>12} "
        f"{total_fdr:>8} {total_fdr - total_raw:>+6}"
    )

    if deflation:
        print(
            "\nDeflación de Yates: phi_yates/phi_sin mediana="
            f"{deflation['median']:.3f} [min {deflation['min']:.3f}, "
            f"max {deflation['max']:.3f}] sobre {deflation['n']} filas"
        )

    print(f"\nphi sin Yates — máximos por clase (top {top}, phi > 0)")
    for cluster in per_class:
        positive = [
            r for r in rows if r["cluster"] == int(cluster) and r["phi_sin_yates"] > 0
        ]
        positive.sort(key=lambda r: r["phi_sin_yates"], reverse=True)
        n_uce = meta["class_sizes"].get(int(cluster), 0)
        print(f"\n  clúster {cluster} ({n_uce} UCE)")
        print(f"    {'término':<24} {'phi_sin':>8} {'phi_yat':>8} {'FDR':>4}")
        for row in positive[:top]:
            mark = "sí" if row["significativo_fdr"] else "no"
            print(
                f"    {str(row['termino'])[:24]:<24} {row['phi_sin_yates']:>8.3f} "
                f"{row['phi']:>8.3f} {mark:>4}"
            )


# ── F9: plano factorial ────────────────────────────────────────────────────


def write_figure(artifact: dict, rows, out_path: Path, top: int) -> bool:
    """Biplot F9 con nivel de presencia e inercia sobre el total."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib no está instalado; no se genera la figura")
        return False

    mv = artifact.get("multivariate") or {}
    row_coords = mv.get("row_coords") or []
    labels = mv.get("cluster_labels") or []
    raw_cols = mv.get("col_coords") or []
    vocab = mv.get("terms") or []
    info = mca_meta(artifact)
    eigen = info["eigenvalues"]

    if not row_coords or not raw_cols or len(eigen) < 2:
        print("[!] el artefacto no trae row_coords/col_coords/eigenvalues")
        return False

    n_terms = len(vocab)
    if len(raw_cols) == 2 * n_terms:
        col_coords = [raw_cols[i] for i in range(1, len(raw_cols), 2)]
    else:
        col_coords = raw_cols

    # Coordenadas estándar -> principales (escala comparable con las filas).
    col_coords = [
        [x * math.sqrt(eigen[0]), y * math.sqrt(eigen[1])] for x, y in col_coords
    ]

    per_class = summarise(rows)
    clusters = [int(k) for k in per_class]
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    colors = {k: palette[i % len(palette)] for i, k in enumerate(clusters)}

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for k in clusters:
        idx = [i for i, lbl in enumerate(labels) if int(lbl) == k]
        if not idx:
            continue
        ax.scatter(
            [row_coords[i][0] for i in idx],
            [row_coords[i][1] for i in idx],
            s=16,
            alpha=0.55,
            linewidths=0,
            color=colors[k],
            label=f"Clase {k} (n={len(idx)})",
        )

    ax.scatter(
        [p[0] for p in col_coords],
        [p[1] for p in col_coords],
        s=4,
        alpha=0.45,
        linewidths=0,
        color="0.45",
        label=f"Términos (n={len(col_coords)})",
    )

    index_of = {str(term): i for i, term in enumerate(vocab)}
    for k in clusters:
        positive = [r for r in rows if r["cluster"] == k and r["phi_sin_yates"] > 0]
        positive.sort(key=lambda r: r["phi_sin_yates"], reverse=True)
        for row in positive[:top]:
            pos = index_of.get(str(row["termino"]))
            if pos is None or pos >= len(col_coords):
                continue
            x, y = col_coords[pos]
            ax.annotate(
                str(row["termino"]),
                (x, y),
                fontsize=7,
                color=colors[k],
                xytext=(2, 2),
                textcoords="offset points",
            )

    ax.axhline(0, linewidth=0.7, color="0.7")
    ax.axvline(0, linewidth=0.7, color="0.7")
    pct = [v * 100 for v in info["explained_inertia_corrected"]]
    ax.set_xlabel(f"F1 ({pct[0]:.2f} % de la inercia total)")
    ax.set_ylabel(f"F2 ({pct[1]:.2f} % de la inercia total)")
    ax.set_title("Plano factorial de la partición de REII")
    ax.legend(loc="best", fontsize=8, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"}
    if out_path.suffix.lower() == ".png":
        kwargs["dpi"] = 200
    fig.savefig(out_path, **kwargs)
    plt.close(fig)
    return True


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> ArgumentParser:
    ap = ArgumentParser(
        prog="reii.diag term-metrics",
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--artifact",
        "--workflow",
        dest="artifact",
        default=str(DEFAULT_ARTIFACT),
        help="artefacto de entrada (default: data/workflow_data.json)",
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="JSON de salida (default: data/diag/paper_terms.json)",
    )
    ap.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="umbral de p_adj para significativo_fdr (default: 0.05)",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=6,
        help="términos por clase a mostrar/etiquetar (default: 6)",
    )
    ap.add_argument(
        "--figure",
        default=None,
        help="ruta .svg o .png del plano factorial F9 (opcional)",
    )
    return ap


def main(argv=None) -> int:
    utf8_stdout()
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"[X] no existe el artefacto: {artifact_path}")
        return 1
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[X] no se pudo leer {artifact_path}: {exc}")
        return 1

    rows, meta = derive_terms(artifact, args.fdr_alpha)
    if not rows:
        print("[X] no se pudo derivar ninguna fila de términos")
        return 1

    worst = max(meta["max_abs_diff_chi2_yates"], meta["max_abs_diff_phi"])
    if worst > VERIFY_TOL:
        print(
            "[X] la reconstrucción 2x2 no cuadra con el artefacto "
            f"(Δmax={worst:.3g} > {VERIFY_TOL:.0e}).\n"
            "    Revisa que el artefacto sea de la corrida con clase asignada."
        )
        return 1

    info = mca_meta(artifact)
    per_class = summarise(rows)
    deflation = yates_deflation(rows)
    print_report(artifact_path, meta, info, per_class, deflation, rows, args.top)

    payload = {
        "artifact": str(artifact_path),
        "fdr_alpha": args.fdr_alpha,
        **meta,
        "mca": info,
        "per_class": per_class,
        "yates_deflation": deflation,
        "terms": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nJSON: {out_path}")

    if args.figure:
        fig_path = Path(args.figure)
        if write_figure(artifact, rows, fig_path, args.top):
            print(f"figura: {fig_path}")
        else:
            return 1
    return 0


def run(argv=None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
