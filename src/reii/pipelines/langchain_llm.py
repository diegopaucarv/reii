"""Wrapper LangChain para el LLM (Together API, compatible con OpenAI).

Expone el modelo como un :class:`~langchain_core.language_models.chat_models.BaseChatModel`
para que el orquestador multiagente (``discourse_graph``) y los agentes
(``discourse_agents``) dependan de una abstracción inyectable en lugar de la
función global ``call_deepseek_structured``. Esto permite probar todo el
sistema **offline** inyectando un modelo falso (p. ej.
``GenericFakeChatModel`` o un stub con ``invoke_structured``).

Dos modos de uso:

* ``invoke_structured(messages, schema)`` → dict validado contra ``schema``
  (reutiliza ``call_deepseek_structured``, con tool-calling y reparación de
  JSON). Es el camino que usan el agente explorador y el orquestador.
* ``_generate`` → completado de texto plano (para llamadas no estructuradas).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from reii.config import (
    LLM_FALLBACK_MODEL,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TIMEOUT,
    TOGETHER_API_KEY,
    TOGETHER_API_URL,
)

#: Mapeo de roles de LangChain a roles de la API OpenAI-compatible.
_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}


def _to_openai_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, str]]:
    """Convierte mensajes de LangChain al formato ``[{role, content}]``."""
    out: List[Dict[str, str]] = []
    for m in messages:
        role = _ROLE_MAP.get(m.type, "user")
        content = m.content if isinstance(m.content, str) else str(m.content)
        out.append({"role": role, "content": content})
    return out


class TogetherChatModel(BaseChatModel):
    """Modelo de chat LangChain respaldado por la API de Together.

    Los campos ``model``, ``temperature``, etc. son configurables para poder
    construir instancias de prueba sin tocar las variables de entorno.
    """

    model: str = LLM_MODEL
    fallback_model: str = LLM_FALLBACK_MODEL
    temperature: float = 0.2
    max_retries: int = LLM_MAX_RETRIES
    timeout: int = LLM_TIMEOUT
    api_key: str = TOGETHER_API_KEY
    base_url: str = TOGETHER_API_URL

    @property
    def _llm_type(self) -> str:
        return "together"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Completado de texto plano (no estructurado)."""
        import requests

        payload = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            self.base_url, headers=headers, json=payload, timeout=(30, self.timeout)
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=_ai_message(content))])

    def invoke_structured(
        self, messages: Sequence[BaseMessage], schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Salida estructurada validada contra ``schema`` (tool-calling).

        Reutiliza ``call_deepseek_structured`` (streaming + reparación de JSON +
        validación de esquema). Devuelve el dict parseado.
        """
        from reii.ia_discursiva import call_deepseek_structured

        return call_deepseek_structured(
            _to_openai_messages(messages),
            schema,
            max_retries=self.max_retries,
            temperature=self.temperature,
        )


def _ai_message(content: str) -> BaseMessage:
    """Construye un ``AIMessage`` sin importar el paquete completo de langchain."""
    from langchain_core.messages import AIMessage

    return AIMessage(content=content)
