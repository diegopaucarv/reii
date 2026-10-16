"""Pruebas offline del modelo de dependencias entre workflows, capas y agentes."""

import json

import pytest

from reii.pipelines.dependency_graph import (
    AgentSpec,
    WorkflowSpec,
    build_workflow_spec,
    load_workflow_specs,
    validate_dependencies,
    workflow_name_from_filename,
)


# ── workflow_name_from_filename ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "filename, expected",
    [
        ("discurso_cohesion.json", "cohesion"),
        ("discurso_critico.json", "critico"),
        ("discurso_lexico.json", "lexico"),
    ],
)
def test_workflow_name_from_filename(filename, expected):
    assert workflow_name_from_filename(filename) == expected


def test_workflow_name_from_filename_handles_paths_and_no_prefix():
    # Con ruta completa y sin prefijo "discurso_".
    assert (
        workflow_name_from_filename("ia/analisis_discursivo/discurso_cohesion.json")
        == "cohesion"
    )
    assert workflow_name_from_filename("otro.json") == "otro"


# ── build_workflow_spec ─────────────────────────────────────────────────────
def _agent(name, disc_cats=(), gram_cats=()):
    return {
        "name": name,
        "role": "rol",
        "instructions": "instrucciones",
        "gram_cats": list(gram_cats),
        "disc_cats": list(disc_cats),
    }


def test_build_workflow_spec_derives_layers_with_longest_path():
    agents = [
        _agent("a"),  # sin deps -> capa 1
        _agent("b"),  # sin deps -> capa 1
        _agent("c", disc_cats=["a"]),  # depende de capa 1 -> capa 2
        _agent("d", disc_cats=["a", "b"]),  # depende de capa 1 -> capa 2
        _agent("e", disc_cats=["c", "d"]),  # depende de capa 2 -> capa 3
    ]
    spec = build_workflow_spec("test", agents)

    assert spec.n_layers == 3
    assert spec.layer_of("a") == 1
    assert spec.layer_of("b") == 1
    assert spec.layer_of("c") == 2
    assert spec.layer_of("d") == 2
    assert spec.layer_of("e") == 3


def test_build_workflow_spec_omits_placeholders():
    agents = [
        _agent("a"),
        {
            "name": "dssf",
            "role": "",
            "instructions": "",
            "gram_cats": [],
            "disc_cats": [],
        },
        {"name": "sin_role_ni_instructions"},  # placeholder sin role ni instructions
    ]
    spec = build_workflow_spec("test", agents)
    assert [a.name for a in spec.agents] == ["a"]


def test_build_workflow_spec_topological_order_respects_layers():
    agents = [
        _agent("e", disc_cats=["c"]),
        _agent("c", disc_cats=["a"]),
        _agent("a"),
    ]
    spec = build_workflow_spec("test", agents)
    order = spec.topological_order()
    # "a" (capa 1) debe ir antes que "c" (capa 2), y "c" antes que "e" (capa 3).
    assert order.index("a") < order.index("c") < order.index("e")


def test_build_workflow_spec_uses_workflow_meta_display_name():
    spec = build_workflow_spec("cohesion", [_agent("a")])
    assert spec.display_name == "Cohesión"
    assert spec.description != ""


# ── WorkflowSpec consultas ──────────────────────────────────────────────────
def test_workflow_spec_layer_of_dependencies_agents_in_layer():
    agents = [
        _agent("a"),
        _agent("b", disc_cats=["a"]),
        _agent("c", disc_cats=["a", "b"]),
    ]
    spec = build_workflow_spec("test", agents)

    assert spec.layer_of("a") == 1
    assert spec.layer_of("b") == 2
    assert spec.layer_of("c") == 3
    assert spec.layer_of("inexistente") is None

    assert spec.dependencies("c") == {"a", "b"}
    assert spec.dependencies("a") == set()
    assert spec.dependencies("inexistente") == set()

    assert [a.name for a in spec.agents_in_layer(1)] == ["a"]
    assert [a.name for a in spec.agents_in_layer(2)] == ["b"]
    assert [a.name for a in spec.agents_in_layer(3)] == ["c"]
    assert spec.agents_in_layer(0) == ()
    assert spec.agents_in_layer(4) == ()


# ── load_workflow_specs ─────────────────────────────────────────────────────
def test_load_workflow_specs_from_tmp_dir(tmp_path):
    # Un workflow válido (dict de agentes).
    (tmp_path / "discurso_cohesion.json").write_text(
        json.dumps(
            {
                "a": _agent("a"),
                "b": _agent("b", disc_cats=["a"]),
            }
        ),
        encoding="utf-8",
    )
    # El resumidor gramatical: dict con "name" string a nivel superior.
    (tmp_path / "resumidor_gramatical.json").write_text(
        json.dumps(
            {"name": "grammatical_summarizer", "role": "r", "instructions": "i"}
        ),
        encoding="utf-8",
    )
    # Archivo no JSON y archivo prefijado con "_" deben ignorarse.
    (tmp_path / "notas.txt").write_text("hola", encoding="utf-8")
    (tmp_path / "_borrador.json").write_text(
        json.dumps({"a": _agent("a")}), encoding="utf-8"
    )

    specs = load_workflow_specs(str(tmp_path))

    assert list(specs) == ["cohesion"]
    assert specs["cohesion"].n_layers == 2


def test_load_workflow_specs_returns_empty_for_missing_dir(tmp_path):
    assert load_workflow_specs(str(tmp_path / "no_existe")) == {}


# ── validate_dependencies ───────────────────────────────────────────────────
def test_validate_dependencies_reports_broken_dependency():
    agents = [_agent("a", disc_cats=["fantasma"])]
    spec = build_workflow_spec("test", agents)
    warnings = validate_dependencies({"test": spec})
    assert len(warnings) == 1
    assert "fantasma" in warnings[0]


def test_validate_dependencies_empty_when_all_ok():
    agents = [_agent("a"), _agent("b", disc_cats=["a"])]
    spec = build_workflow_spec("test", agents)
    assert validate_dependencies({"test": spec}) == []


# ── integración con el directorio real ──────────────────────────────────────
def test_load_real_workflow_specs_layers_and_no_warnings():
    from reii.config import DISCOURSE_DIR

    specs = load_workflow_specs(str(DISCOURSE_DIR))

    assert set(specs) == {"cohesion", "critico", "lexico"}
    assert specs["cohesion"].n_layers == 4
    assert specs["critico"].n_layers == 3
    assert specs["lexico"].n_layers == 5
    assert validate_dependencies(specs) == []
