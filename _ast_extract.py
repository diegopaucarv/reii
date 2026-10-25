"""Utilidad compartida para los tests de este directorio (no es un test en
sí — pytest no la recoge porque no matchea test_*.py / *_test.py).

Extrae una función o método de main_workflow.py por nombre, vía AST, y lo
ejecuta en un namespace mínimo con solo las dependencias que esa función en
particular necesita. Así se puede probar la lógica REAL del archivo (no una
reimplementación aparte que podría desincronizarse) sin necesitar instalado
todo requirements.txt (nltk, spacy, prince, transformers, torch, hdbscan,
stanza...) — basta con lo que cada función puntual importa.

Mismo patrón que ya usa tests/test_optimizer_budget_bugs.py para
_perturb_params (ver _load_perturb ahí) — esto solo lo generaliza para
reusarlo en varios archivos de test.
"""
import ast
import textwrap
from pathlib import Path

MAIN_WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "reii" / "main_workflow.py"
)


def extract_function(name: str, namespace: dict | None = None):
    """Busca `def {name}(...)` (función top-level o método) en
    main_workflow.py, lo extrae con ast.get_source_segment y lo ejecuta en
    `namespace` (más builtins). Devuelve el callable ya definido.

    `namespace` debe traer cualquier dependencia que el cuerpo de la función
    referencie como variable libre (otro import de módulo, una constante
    módulo-level, un helper). Si falta algo, el exec() revienta con
    NameError — así es evidente qué agregar, no falla en silencio.

    Nota: para un método (primer parámetro `self`), el callable resultante
    sigue pidiendo `self` como primer argumento posicional explícito; se le
    pasa un stub mínimo (ver ejemplos en los tests de este directorio).
    """
    src_text = MAIN_WORKFLOW_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src_text)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            seg = ast.get_source_segment(src_text, node)
            ns = dict(namespace or {})
            exec(textwrap.dedent(seg), ns)
            return ns[name]
    raise AssertionError(
        f"'{name}' no se encontró como función/método en {MAIN_WORKFLOW_PATH}. "
        "¿Se renombró? Si es intencional, actualiza el test correspondiente."
    )
