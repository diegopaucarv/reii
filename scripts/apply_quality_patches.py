#!/usr/bin/env python3
"""Aplica (o revierte) los tres parches mínimos que conectan reii.quality con el repo.

  workflow_hook : tras ``self._save_all_uces(...)`` guarda la corrida de calidad.
  mapping_env   : ``_hungarian_stability_direct`` delega en reii.quality.mapping si
                  REII_STABILITY_MAPPING=many_to_one (por defecto sigue "hungarian").
  dashboard_tab : añade la pestaña "F · Calidad y niveles" a dashboard.py.

Uso:  python scripts/apply_quality_patches.py [--repo RUTA] [--dry-run] [--revert]
Es idempotente (marca [quality-patch]) y crea copias .bak antes de escribir.
Respeta los finales de línea (CRLF/LF) de cada archivo.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

MARK = "[quality-patch]"


def _read(path):
    raw = open(path, "rb").read().decode("utf-8")
    crlf = "\r\n" in raw
    return raw.replace("\r\n", "\n"), crlf


def _write(path, text, crlf):
    if crlf:
        text = text.replace("\n", "\r\n")
    open(path, "wb").write(text.encode("utf-8"))


def patch_workflow_hook(text):
    anchor = "self._save_all_uces(uces_est_list, uces_por_doc)"
    if MARK + " hook" in text:
        return text, "ya aplicado"
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if anchor in l:
            ind = l[: len(l) - len(l.lstrip())]
            block = [
                f"{ind}try:  # {MARK} hook",
                f"{ind}    from reii.quality.hooks import save_quality_run",
                "",
                f"{ind}    save_quality_run(self, uces_est_list, uces_por_doc)",
                f"{ind}except Exception as _qe:  # nunca rompe el pipeline",
                f'{ind}    logger.warning("quality hook falló: %s", _qe)',
            ]
            lines[i + 1 : i + 1] = block
            return "\n".join(lines), "aplicado"
    return text, "ANCLA NO ENCONTRADA"


def patch_mapping_env(text):
    if MARK + " mapping" in text:
        return text, "ya aplicado"
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if "def _hungarian_stability_direct(" in l:
            j = i
            while "linear_sum_assignment is None" not in lines[j]:
                j += 1
            ind = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
            block = [
                f'{ind}_q_strategy = os.environ.get("REII_STABILITY_MAPPING", "hungarian")  # {MARK} mapping',
                f'{ind}if _q_strategy != "hungarian":',
                f"{ind}    from reii.quality.mapping import hungarian_compatible",
                "",
                f"{ind}    return hungarian_compatible(uce_to_ca, uce_to_cb, strategy=_q_strategy, label=label)",
            ]
            lines[j:j] = block
            return "\n".join(lines), "aplicado"
    return text, "ANCLA NO ENCONTRADA"


def patch_dashboard_tab(text):
    if MARK + " tab" in text:
        return text, "ya aplicado"
    old_head = "tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs("
    old_item = '        "E · Discurso por clase",\n'
    if old_head not in text or old_item not in text:
        return text, "ANCLA NO ENCONTRADA (pestañas)"
    text = text.replace(old_head, "tab_a, tab_b, tab_c, tab_d, tab_e, tab_f = st.tabs(  # " + MARK + " tab", 1)
    text = text.replace(old_item, old_item + '        "F · Calidad y niveles",\n', 1)
    anchor = "# ── Sidebar: botones de salto (JMP)"
    if anchor not in text:
        return text, "ANCLA NO ENCONTRADA (sidebar)"
    block = (
        "if tab_f.open:\n"
        "    with tab_f:\n"
        "        try:\n"
        "            from reii.quality.dashboard_panel import render_quality_panel\n\n"
        "            render_quality_panel()\n"
        "        except Exception as _qe:\n"
        '            st.error(f"Panel de calidad no disponible: {_qe}")\n\n'
    )
    return text.replace(anchor, block + anchor, 1), "aplicado"


PATCHES = [
    ("src/reii/main_workflow.py", "workflow_hook", patch_workflow_hook),
    ("src/reii/main_workflow.py", "mapping_env", patch_mapping_env),
    ("src/reii/dashboard.py", "dashboard_tab", patch_dashboard_tab),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true", help="restaura los .bak")
    a = ap.parse_args()
    if a.revert:
        for rel in sorted({p[0] for p in PATCHES}):
            path, bak = os.path.join(a.repo, rel), os.path.join(a.repo, rel) + ".quality.bak"
            if os.path.exists(bak):
                shutil.copy2(bak, path)
                print(f"restaurado {rel}")
        return 0
    status = 0
    cache = {}
    for rel, name, fn in PATCHES:
        path = os.path.join(a.repo, rel)
        if rel not in cache:
            cache[rel] = _read(path)
        text, crlf = cache[rel]
        new, msg = fn(text)
        print(f"{name:14s} {rel}: {msg}")
        if msg.startswith("ANCLA"):
            status = 1
        cache[rel] = (new, crlf)
    if not a.dry_run and status == 0:
        for rel, (text, crlf) in cache.items():
            path = os.path.join(a.repo, rel)
            bak = path + ".quality.bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            _write(path, text, crlf)
    return status


if __name__ == "__main__":
    sys.exit(main())
