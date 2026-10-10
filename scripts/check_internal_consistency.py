"""
Diagnóstico 3: consistencia interna por doc.

Para cada UCE verifica:
  1. ¿Las menciones de coref_chains EMPIEZAN dentro de [start_char, end_char)?
     (start_char/end_char de las menciones son GLOBALES; el fin puede
     cruzar el límite de la UCE — el segmentador usa ventana de coref
     (uc_window_size) precisamente porque las menciones cruzan UCEs)
  2. ¿Los predicate_frames caen dentro de [start_char, end_char]?
     (entity/verb/object offsets son GLOBALES; las expansiones usan
     verb_start_char=0, verb_end_char=0 como placeholder y se omiten)
  3. ¿Los frame_annotations caen dentro de [0, len(texto)]?
     (char_start/char_end de las anotaciones son LOCALES a la UCE;
     char_end puede ser == len(texto), es un rango exclusivo)

Régimen de offsets:
  - coref_chains.mentions[].start_char/end_char  → GLOBAL
  - predicate_frames[].entity_start_char/entity_end_char,
    verb_start_char/verb_end_char, direct_object_start/end → GLOBAL
  - frame_annotations[].char_start/char_end       → LOCAL (0..len(texto))

Si un doc es internamente consistente, el dashboard (que resta uce.start_char)
renderiza bien aunque el régimen de offsets difiera entre docs.
"""

import os
import sys

import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get("REII_DATABASE_URL", "postgresql://reii:reii@reii-db:5432/reii")


def _in_range(value, lo, hi, end_inclusive: bool = False) -> bool:
    """True si lo <= value < hi (o <= hi si end_inclusive). None se ignora."""
    if value is None:
        return True
    if end_inclusive:
        return lo <= value <= hi
    return lo <= value < hi


def main() -> None:
    conn = psycopg.connect(DSN, row_factory=dict_row)
    rows = conn.execute(
        """
        SELECT doc_id, local_idx, texto,
               (linguistic_json::jsonb->>'start_char')::int AS start_char,
               (linguistic_json::jsonb->>'end_char')::int AS end_char,
               linguistic_json::jsonb->'coref_chains' AS coref_chains,
               linguistic_json::jsonb->'predicate_frames' AS predicate_frames,
               linguistic_json::jsonb->'frame_annotations' AS frame_annotations
        FROM uces
        ORDER BY doc_id, local_idx
        """
    ).fetchall()

    by_doc: dict = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)

    total_coref_out = 0
    total_frame_out = 0
    total_ann_out = 0

    print(f"{'doc':>4} {'UCEs':>5} {'coref_out':>9} {'frame_out':>9} {'ann_out':>7}")
    for doc_id in sorted(by_doc, key=lambda d: int(d)):
        doc_uces = by_doc[doc_id]
        coref_out = 0
        frame_out = 0
        ann_out = 0
        for u in doc_uces:
            sc, ec = u["start_char"], u["end_char"]
            if sc is None or ec is None:
                continue
            txt_len = len(u["texto"] or "")

            # 1. coref_chains: GLOBALES. Solo se exige que la mención EMPIECE
            #    dentro de la UCE; el fin puede cruzar al siguiente segmento
            #    (ventana de coref del segmentador).
            for ch in u["coref_chains"] or []:
                for m in ch.get("mentions", []):
                    if not _in_range(m.get("start_char"), sc, ec):
                        coref_out += 1

            # 2. predicate_frames: GLOBALES. Entidad siempre en rango; verbo
            #    solo para frames base (las expansiones usan 0,0 placeholder);
            #    objeto directo en rango cuando existe.
            for pf in u["predicate_frames"] or []:
                is_exp = pf.get("is_expansion") is True
                if not _in_range(pf.get("entity_start_char"), sc, ec) or not _in_range(
                    pf.get("entity_end_char"), sc, ec, end_inclusive=True
                ):
                    frame_out += 1
                    continue
                if not is_exp and (
                    not _in_range(pf.get("verb_start_char"), sc, ec)
                    or not _in_range(
                        pf.get("verb_end_char"), sc, ec, end_inclusive=True
                    )
                ):
                    frame_out += 1
                    continue
                if not _in_range(
                    pf.get("direct_object_start"), sc, ec
                ) or not _in_range(
                    pf.get("direct_object_end"), sc, ec, end_inclusive=True
                ):
                    frame_out += 1

            # 3. frame_annotations: LOCALES → [0, len(texto)].
            #    char_end puede ser == len(texto) (rango exclusivo).
            for ann in u["frame_annotations"] or []:
                if not _in_range(ann.get("char_start"), 0, txt_len) or not _in_range(
                    ann.get("char_end"), 0, txt_len, end_inclusive=True
                ):
                    ann_out += 1

        total_coref_out += coref_out
        total_frame_out += frame_out
        total_ann_out += ann_out
        print(
            f"{doc_id:>4} {len(doc_uces):>5} {coref_out:>9} {frame_out:>9} {ann_out:>7}"
        )

    print("-" * 40)
    print(
        f"TOTAL  coref_out={total_coref_out}  frame_out={total_frame_out}  "
        f"ann_out={total_ann_out}"
    )
    conn.close()

    if total_coref_out or total_frame_out or total_ann_out:
        sys.exit(1)


if __name__ == "__main__":
    main()
