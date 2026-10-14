"""Tests de la carga modular del pipeline de Stanza (``build_stanza_pipeline``).

El módulo ``reii.seg.stanza_pipeline`` importa ``stanza`` de forma perezosa, así
que estos tests son rápidos: inyectan un falso ``stanza`` parcheando
``_get_stanza_module`` y no arrastran torch/spacy/sentence_transformers.
"""

from unittest import mock

from reii.seg.stanza_pipeline import COREF_PROCESSORS, build_stanza_pipeline


def _make_fake_stanza():
    fake = mock.Mock()
    fake.download = mock.Mock()
    fake.Pipeline = mock.Mock()
    return fake


def test_builds_pipeline_with_requested_processors():
    fake = _make_fake_stanza()
    pipe = object()
    fake.Pipeline.return_value = pipe
    with mock.patch("reii.seg.stanza_pipeline._get_stanza_module", return_value=fake):
        result = build_stanza_pipeline("es", COREF_PROCESSORS, use_gpu=False)

    fake.download.assert_called_once_with(
        "es", processors=COREF_PROCESSORS, verbose=False
    )
    fake.Pipeline.assert_called_once_with(
        "es", processors=COREF_PROCESSORS, use_gpu=False, verbose=False
    )
    assert result is pipe


def test_gpu_fallback_to_cpu():
    fake = _make_fake_stanza()
    fake.Pipeline.side_effect = [RuntimeError("OOM"), "cpu_pipe"]
    with mock.patch("reii.seg.stanza_pipeline._get_stanza_module", return_value=fake):
        result = build_stanza_pipeline("es", COREF_PROCESSORS, use_gpu=True)

    assert result == "cpu_pipe"
    assert fake.Pipeline.call_count == 2


def test_returns_none_on_failure():
    fake = _make_fake_stanza()
    fake.Pipeline.side_effect = RuntimeError("boom")
    with mock.patch("reii.seg.stanza_pipeline._get_stanza_module", return_value=fake):
        result = build_stanza_pipeline("es", COREF_PROCESSORS, use_gpu=False)

    assert result is None
    fake.download.assert_called_once_with(
        "es", processors=COREF_PROCESSORS, verbose=False
    )


def test_coref_processors_include_coref():
    assert "coref" in COREF_PROCESSORS.split(",")
