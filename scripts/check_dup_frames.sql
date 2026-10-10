-- Detección de frames duplicados por doc (misma entidad + verbo + offsets)
-- Clave estricta: (entity_start_char, entity_end_char, verb_start_char,
-- verb_end_char, is_expansion). Mismo span de entidad + mismo span de verbo
-- = mismo evento. Tras dedupe_frames.py debe reportar 0 duplicados.
WITH frames AS (
  SELECT doc_id, uce_id, f
  FROM uces, jsonb_array_elements(linguistic_json::jsonb->'predicate_frames') AS f
)
SELECT doc_id,
       count(*) AS n_frames,
       count(*) FILTER (WHERE f->>'is_expansion' = 'true') AS n_expansions,
       count(*) - count(DISTINCT (f->>'entity_start_char', f->>'entity_end_char',
                                  f->>'verb_start_char', f->>'verb_end_char',
                                  f->>'is_expansion')) AS dup_frames
FROM frames
GROUP BY doc_id
ORDER BY doc_id;
