"""Cliente HTTP del coref remoto (``/v1/extract_coref``).

El servidor Stanza de Colab (``colab/stanza_server.py``) devuelve, por texto,
las cadenas de correferencia con offsets **relativos a ese texto** y sin
``representative`` (solo menciones). Este módulo las normaliza al formato que
espera ``gramatical_analyzer.CoreferenceResolver`` (``representative`` +
``mentions`` con offsets locales), para que el pipeline gramatical resuelva el
coref sin cargar un pipeline Stanza local.

Reproduce la petición de ``ProgressiveSegmenter._extract_coref_remote``, pero
separado en su propio módulo para no arrastrar torch/spacy al importarlo.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class RemoteCorefClient:
    """Cliente mínimo del endpoint ``/v1/extract_coref``."""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        lang: str = "es",
        timeout: float = 90.0,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.token = token or ""
        self.lang = lang
        self.timeout = float(timeout)

    def chains_for_texts(self, texts: List[str]) -> List[List[Dict]]:
        """Cadenas por texto, en el mismo orden que ``texts``.

        Los offsets de cada mención son locales al texto enviado. Lanza si el
        servidor no responde o devuelve un número de resultados distinto al de
        textos; el llamador decide cómo degradar.
        """
        if not texts:
            return []

        import requests  # import perezoso: el módulo queda ligero de importar

        resp = requests.post(
            f"{self.base_url}/v1/extract_coref",
            json={"texts": texts, "lang": self.lang},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if len(results) != len(texts):
            raise ValueError(
                f"el servidor devolvió {len(results)} resultados para "
                f"{len(texts)} textos"
            )
        return [self._normalize(r) for r in results]

    @staticmethod
    def _normalize(result: Dict) -> List[Dict]:
        chains_out: List[Dict] = []
        for chain in result.get("chains", []):
            mentions = []
            for m in chain.get("mentions", []):
                start = m.get("start_char")
                end = m.get("end_char")
                if start is None or end is None:
                    continue
                mentions.append(
                    {
                        "text": m.get("text", ""),
                        "start_char": int(start),
                        "end_char": int(end),
                    }
                )
            if not mentions:
                continue
            mentions.sort(key=lambda x: x["start_char"])
            chains_out.append(
                {"representative": _representative(mentions), "mentions": mentions}
            )
        return chains_out


def _representative(mentions: List[Dict]) -> str:
    """Mención más descriptiva de la cadena (texto más largo; empate: la 1.ª).

    El servidor no envía ``representative_text``. Dejarlo vacío rompería el
    agrupado por cadena de ``CorefPredicateAnalyzer`` (usa el representante como
    clave), así que se aproxima con la mención más larga: determinista y estable
    entre ventanas, que es lo que necesita la fusión de cadenas.
    """
    best = ""
    for m in mentions:
        text = (m.get("text") or "").strip()
        if len(text) > len(best):
            best = text
    return best
