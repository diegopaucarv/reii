import json

from reii.gram.gramatical_analyzer import UCE
from reii.main_workflow import Config, SegmentadorALCESTE

config = Config()
seg = SegmentadorALCESTE(config)

# ── Custom grammar values (set directly on the resolved instance) ──
seg.stopwords_extra = {"climático"}  # ADJ -> supplementary
seg.expresiones = ["cambio climático"]  # multi-word -> joined
seg.palabras_activas = {"una": "DET"}  # DET -> forced active
seg.palabras_suplementarias = {"canción"}  # NOUN -> forced supplementary
seg.palabras_excluir = {"paciente"}  # drop entirely
seg.sinonimos = {"cantaba": "cantar"}  # variant -> canonical
seg.sinonimos = {"cantaba": "cantar"}        # variant -> canonical

uce = UCE(id="test0", texto="El paciente cantaba una canción de cambio climático")
seg.lematizar_uces([uce])

print("lemmas:", uce.lemmas)
print("stems:", uce.stems)
print("pos_tags:", uce.pos_tags)
print("marcadores:", uce.marcadores)
print("tokens:", uce.tokens)
print("formas_tokens:", json.dumps(uce.formas_tokens, ensure_ascii=False))
