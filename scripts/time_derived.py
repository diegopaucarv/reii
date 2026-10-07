"""Break down the module-level derived cost: discourse merge vs lemma_map vs rest."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402  (pre-import so it's not counted)

from reii.backend.dashboard_adapter import DashboardAdapter  # noqa: E402

DB = "data/workflow_data.db"
DISC = "data/discourse_state.json"

adapter = DashboardAdapter(DB)
t0 = time.perf_counter()
snap = adapter.snapshot()
t_snap = time.perf_counter() - t0
print(f"snapshot(): {t_snap:.2f}s")

uces = snap.get("uces", [])
print(f"  uces: {len(uces)}")

# Discourse merge (L1029-1049 equivalent)
t0 = time.perf_counter()
with open(DISC, "r", encoding="utf-8") as f:
    disc = json.load(f)
annotations_by_uce = disc.get("annotations_by_uce", {})
if annotations_by_uce:
    for uce in uces:
        raw_anns = annotations_by_uce.get(uce.get("id"), [])
        normalized = []
        for ann in raw_anns:
            if "spans" not in ann and ann.get("uce_id") and ann.get("quote"):
                ann = {
                    **ann,
                    "spans": [
                        {
                            "uce_id": ann["uce_id"],
                            "quote": ann["quote"],
                            "start_char": ann.get("start_char", -1),
                            "end_char": ann.get("end_char", -1),
                        }
                    ],
                }
            normalized.append(ann)
        uce["discourse_annotations"] = normalized
t_disc = time.perf_counter() - t0
print(f"discourse merge: {t_disc:.3f}s")

# lemma_map build (L1336-1350 equivalent)
t0 = time.perf_counter()
from collections import defaultdict

lemma_map = defaultdict(lambda: {"stems": set(), "formas": set(), "total_freq": 0})
_forma_idx = snap.get("forma_index", {})
for cid, stems in _forma_idx.items():
    for stem, lemmas in stems.items():
        for lemma, formas in lemmas.items():
            if lemma not in lemma_map:
                lemma_map[lemma] = {"stems": set(), "formas": set(), "total_freq": 0}
            lemma_map[lemma]["stems"].add(stem)
            if isinstance(formas, dict):
                for f_exact, count in formas.items():
                    lemma_map[lemma]["formas"].add(f_exact)
                    lemma_map[lemma]["total_freq"] += count
            else:
                lemma_map[lemma]["formas"].add(lemma)
lemma_map = dict(lemma_map)
t_lemma = time.perf_counter() - t0
print(f"lemma_map build: {t_lemma:.3f}s ({len(lemma_map)} lemmas)")

# uce_phi_dict + term_stability + clusters + class_sizes (cheap ones)
t0 = time.perf_counter()
uce_phi = snap.get("uce_phi", [])
uce_phi_dict = {item["uce_id"]: item["phi_score"] for item in uce_phi}
term_stability_raw = snap.get("term_stability", [])
term_stability_dict = {}
for _ts in term_stability_raw:
    _key = (str(_ts.get("termino", "")), int(_ts.get("cluster", -1)))
    term_stability_dict[_key] = float(_ts.get("selection_freq", 0.0))
from collections import Counter

class_sizes = Counter()
for uce in uces:
    cid = uce.get("cluster_id")
    if cid is not None and cid >= 0:
        class_sizes[cid] += 1
t_cheap = time.perf_counter() - t0
print(f"uce_phi + term_stability + class_sizes: {t_cheap:.3f}s")

print(f"\nTOTAL derived (excl. snapshot): {t_disc + t_lemma + t_cheap:.3f}s")
