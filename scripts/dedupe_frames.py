#!/usr/bin/env python3
"""
dedupe_frames.py — Elimina frames duplicados de data/workflow_data.json.

El bug: CorefPredicateAnalyzer._extract_frames_from_uce itera TODAS las
cadenas de coreferencia × TODOS los mentions. El mismo span de entidad
(entity_start_char, entity_end_char) aparece en VARIAS cadenas, así que el
mismo evento (span de entidad + span de verbo) produce N frames que solo
difieren en chain_representative.

Este script post-procesa el JSON sin re-ejecutar el pipeline (~14h):

  1. Por cada UCE, deduplica predicate_frames con la clave estricta
     (entity_start_char, entity_end_char, verb_start_char, verb_end_char,
     is_expansion). Mismo span de entidad + mismo span de verbo = mismo
     evento, punto.
  2. Entre duplicados se conserva el frame con chain_representative MÁS
     largo (el más descriptivo para clustering); empates → primera
     aparición. Determinista e idempotente.
  3. Regenera frame_annotations replicando SpanAnnotationIndex.
     to_uce_annotations (offsets LOCALES = global - uce.start_char,
     saltando rangos inválidos, ordenados por char_start).
  4. Re-etiqueta frame_idx secuencialmente (igual que extract()).

Solo toca predicate_frames / frame_annotations de cada UCE; el resto del
JSON se conserva byte a byte (mismo formato: indent=2, ensure_ascii=False,
sin BOM).

Uso:
    python scripts/dedupe_frames.py [INPUT] [--output OUTPUT] [--dry-run]

    INPUT    ruta del JSON (default: data/workflow_data.json)
    --output ruta de salida (default: INPUT, es decir, in-place)
    --dry-run  reporta conteos sin escribir nada
"""

import argparse
import json
import os
from pathlib import Path


# Clave de deduplicación: mismo span de entidad + mismo span de verbo
# = mismo evento, aunque la coreferencia lo asigne a varias cadenas.
def _dedupe_key(frame: dict) -> tuple:
    return (
        frame.get("entity_start_char"),
        frame.get("entity_end_char"),
        frame.get("verb_start_char"),
        frame.get("verb_end_char"),
        frame.get("is_expansion"),
    )


def dedupe_frames(frames: list) -> list:
    """Deduplica frames de una UCE por la clave estricta.

    Conserva el frame con chain_representative más largo (más descriptivo
    para clustering); empates → primera aparición. Re-etiqueta frame_idx
    secuencialmente, igual que CorefPredicateAnalyzer.extract().
    """
    best_idx: dict = {}
    best_frame: dict = {}
    deduped: list = []
    for frame in frames:
        key = _dedupe_key(frame)
        if key not in best_idx:
            best_idx[key] = len(deduped)
            best_frame[key] = frame
            deduped.append(frame)
        elif len(frame.get("chain_representative", "")) > len(
            best_frame[key].get("chain_representative", "")
        ):
            deduped[best_idx[key]] = frame
            best_frame[key] = frame
    # Re-etiquetar frame_idx (derivado de la posición en la lista)
    for i, frame in enumerate(deduped):
        frame["frame_idx"] = i
    return deduped


def regenerate_annotations(uce: dict, frames: list) -> list:
    """Replica SpanAnnotationIndex.to_uce_annotations (gramatical_analyzer.py).

    Emite anotaciones ENTITY/VERB/OBJECT con offsets LOCALES
    (global - uce.start_char), saltando rangos inválidos
    (s/e None, s < 0, e > len(texto)). Ordenadas por char_start.
    """
    base = uce.get("start_char") or 0
    texto = uce.get("texto", "")
    anns: list = []

    for f in frames:

        def loc(g):
            return (g - base) if g is not None else None

        local = {
            "entity_start": loc(f.get("entity_start_char")),
            "entity_end": loc(f.get("entity_end_char")),
            "verb_start": loc(f.get("verb_start_char")),
            "verb_end": loc(f.get("verb_end_char")),
            "obj_start": loc(f.get("direct_object_start")),
            "obj_end": loc(f.get("direct_object_end")),
        }

        for start_key, end_key, span_type in (
            ("entity_start", "entity_end", "ENTITY"),
            ("verb_start", "verb_end", "VERB"),
            ("obj_start", "obj_end", "OBJECT"),
        ):
            s = local[start_key]
            e = local[end_key]
            if s is None or e is None or s < 0 or e > len(texto):
                continue
            anns.append(
                {
                    "char_start": s,
                    "char_end": e,
                    "span_type": span_type,
                    "cluster_id": f.get("cluster_id", -1),
                    "cluster_label": f.get("cluster_label", ""),
                    "thematic_role": f.get("thematic_role", "UNSPECIFIED"),
                    "frame_fingerprint": f.get("frame_fingerprint", ""),
                    "chain": f.get("chain_representative", ""),
                    "negated": f.get("negated", False),
                    "voice": f.get("voice", ""),
                }
            )

    return sorted(anns, key=lambda a: a["char_start"])


def verify_format(path: Path) -> None:
    """Verifica el formato del archivo antes de escribir (sin BOM, JSON indentado)."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(60)
    if head.startswith("\ufeff"):
        raise SystemExit(f"ERROR: {path} tiene BOM; no se tocará el archivo.")
    if not head.startswith('{\n  "uces"'):
        raise SystemExit(f"ERROR: formato inesperado al inicio de {path}: {head!r}")
    with open(path, "rb") as f:
        f.seek(-60, os.SEEK_END)
        tail = f.read().decode("utf-8")
    if not tail.rstrip().endswith("}"):
        raise SystemExit(f"ERROR: formato inesperado al final de {path}: {tail!r}")
    print("Formato verificado: sin BOM, indent=2, ensure_ascii=False")
    print(f"  primeros 60 chars: {head!r}")
    print(f"  últimos  60 chars: {tail!r}")


def process(data: dict) -> tuple:
    """Deduplica frames y regenera anotaciones en TODAS las UCEs.

    Devuelve (data_modificada, stats) donde stats es un dict con los
    conteos globales.
    """
    total_before = 0
    total_after = 0
    uces_with_frames = 0
    uces_affected = 0

    for uce in data.get("uces", []):
        if "predicate_frames" not in uce:
            continue
        frames = uce.get("predicate_frames") or []
        total_before += len(frames)
        if frames:
            uces_with_frames += 1
        deduped = dedupe_frames(frames)
        total_after += len(deduped)
        if len(deduped) != len(frames):
            uces_affected += 1
        uce["predicate_frames"] = deduped
        uce["frame_annotations"] = regenerate_annotations(uce, deduped)

    stats = {
        "total_before": total_before,
        "total_after": total_after,
        "dups_removed": total_before - total_after,
        "uces_with_frames": uces_with_frames,
        "uces_affected": uces_affected,
    }
    return data, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplica predicate_frames y regenera frame_annotations "
        "en data/workflow_data.json (sin re-ejecutar el pipeline)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/workflow_data.json",
        help="ruta del JSON (default: data/workflow_data.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="ruta de salida (default: INPUT, es decir, in-place)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="reporta conteos sin escribir nada",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    if not in_path.exists():
        raise SystemExit(f"ERROR: no existe {in_path}")

    print(f"Cargando {in_path} ...")
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data, stats = process(data)

    print(f"UCEs totales:        {len(data.get('uces', []))}")
    print(f"UCEs con frames:      {stats['uces_with_frames']}")
    print(f"Frames antes:         {stats['total_before']}")
    print(f"Frames después:       {stats['total_after']}")
    print(f"Duplicados eliminados: {stats['dups_removed']}")
    print(f"UCEs afectadas:       {stats['uces_affected']}")

    if args.dry_run:
        print("[dry-run] No se escribió nada.")
        return

    if in_path.resolve() == out_path.resolve():
        verify_format(in_path)

    print(f"Escribiendo {out_path} ...")
    # newline="\n" evita que Windows traduzca \n → \r\n al escribir
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Listo.")


if __name__ == "__main__":
    main()
