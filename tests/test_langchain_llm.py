"""Pruebas offline del wrapper LangChain para el LLM (sin red ni LLM)."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from reii.pipelines.langchain_llm import TogetherChatModel, _to_openai_messages


# ── _to_openai_messages ─────────────────────────────────────────────────────
def test_to_openai_messages_maps_roles_and_content():
    messages = [
        SystemMessage(content="eres un asistente"),
        HumanMessage(content="hola"),
        AIMessage(content="¿en qué te ayudo?"),
    ]
    assert _to_openai_messages(messages) == [
        {"role": "system", "content": "eres un asistente"},
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "¿en qué te ayudo?"},
    ]


def test_to_openai_messages_coerces_non_string_content():
    # El contenido no-string se convierte a string.
    messages = [HumanMessage(content=[{"type": "text", "text": "hola"}])]
    out = _to_openai_messages(messages)
    assert out[0]["role"] == "user"
    assert "hola" in out[0]["content"]


# ── TogetherChatModel ───────────────────────────────────────────────────────
def test_together_chat_model_llm_type():
    assert TogetherChatModel()._llm_type == "together"


def test_invoke_structured_returns_fake_dict_and_calls_with_converted_messages(
    monkeypatch,
):
    fake_result = {"annotations": [{"trait": "metáfora"}]}
    calls = {}

    def fake_call_deepseek_structured(
        messages, schema, max_retries=None, temperature=None
    ):
        calls["messages"] = messages
        calls["schema"] = schema
        calls["max_retries"] = max_retries
        calls["temperature"] = temperature
        return fake_result

    monkeypatch.setattr(
        "reii.ia_discursiva.call_deepseek_structured", fake_call_deepseek_structured
    )

    model = TogetherChatModel(max_retries=3, temperature=0.5)
    schema = {"type": "object"}
    messages = [HumanMessage(content="analiza esto")]

    result = model.invoke_structured(messages, schema)

    assert result == fake_result
    assert calls["messages"] == [{"role": "user", "content": "analiza esto"}]
    assert calls["schema"] == schema
    assert calls["max_retries"] == 3
    assert calls["temperature"] == 0.5
