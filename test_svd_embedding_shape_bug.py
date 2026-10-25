"""Regresión: _svd_corpus_embeddings usaba U (vectores de UC) en vez de
Vt.T (vectores de término) al derivar embeddings de término desde el SVD
de ppmi_matrix (n_ucs, n_terms).

Dos efectos del mismo bug, cubiertos aquí por separado:
  1. Cuando n_terms > n_ucs: IndexError al indexar term_embeddings (que solo
     tiene n_ucs filas) con un índice de vocabulario que llega hasta
     n_terms-1. Esto es lo que vio Diego en producción
     ("index 392 is out of bounds for axis 0 with size 392").
  2. Cuando n_terms <= n_ucs: NO truena, pero cada "embedding de término"
     devuelto es en realidad un vector de UC mal etiquetado — corrupción
     silenciosa. Un test que solo mire la forma/el largo del resultado NO
     detecta este segundo caso (ver test_no_crash_but_wrong_content_when_n_terms_le_n_ucs);
     por eso se verifica también el CONTENIDO contra un SVD recalculado
     de forma independiente con la misma semilla (random_state=42, hardcodeada
     en la función real).

Corre la función REAL de main_workflow.py (extraída vía AST, ver
_ast_extract.py) — no una reimplementación aparte que podría desincronizarse
del código fuente. Solo necesita numpy/scipy/scikit-learn (ya en
requirements.txt como dependencias "core"), nada del stack pesado de NLP.
"""
import logging
from typing import Dict, List

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse import random as sparse_random
from sklearn.utils.extmath import randomized_svd

from _ast_extract import extract_function


class _Cfg:
    svd_components = 10


class _StubSelf:
    cfg = _Cfg()


@pytest.fixture(scope="module")
def svd_corpus_embeddings():
    namespace = {
        "np": np,
        "randomized_svd": randomized_svd,
        "logger": logging.getLogger("test"),
        "csr_matrix": csr_matrix,
        "Dict": Dict,
        "List": List,
    }
    return extract_function("_svd_corpus_embeddings", namespace)


def _random_ppmi(n_ucs: int, n_terms: int, seed: int = 0) -> csr_matrix:
    return sparse_random(n_ucs, n_terms, density=0.15, random_state=seed, format="csr")


def test_crashes_or_not_n_terms_greater_than_n_ucs(svd_corpus_embeddings):
    """Caso real de producción: vocab más grande que el número de UCs.
    Con el bug (U en vez de Vt.T), esto lanza IndexError. Con el fix,
    debe devolver una entrada por cada palabra del vocabulario sin tronar.
    """
    n_ucs, n_terms = 50, 80
    mat = _random_ppmi(n_ucs, n_terms)
    vocab = [f"w{i}" for i in range(n_terms)]

    result = svd_corpus_embeddings(_StubSelf(), mat, vocab)

    assert len(result) == n_terms, (
        f"Se esperaban {n_terms} embeddings de término (uno por palabra del "
        f"vocabulario), llegaron {len(result)}. Si esto truena con IndexError "
        "en vez de fallar el assert, el bug U-vs-Vt.T sigue presente."
    )
    assert all(w in result for w in vocab)
    assert all(v.shape == (10,) for v in result.values())


def test_no_crash_but_wrong_content_when_n_terms_le_n_ucs(svd_corpus_embeddings):
    """Caso que NO truena ni con el bug presente (n_terms <= n_ucs) — pero
    el contenido de cada 'embedding de término' debe salir de Vt.T, no de U.
    Un test que solo mirara el largo del resultado pasaría igual con el bug
    activo; por eso se compara contra un SVD recalculado independientemente.
    """
    n_ucs, n_terms = 80, 50
    mat = _random_ppmi(n_ucs, n_terms)
    vocab = [f"w{i}" for i in range(n_terms)]

    result = svd_corpus_embeddings(_StubSelf(), mat, vocab)

    # Mismos parámetros que hardcodea la función real (n_iter=10, seed=42).
    U, Sigma, Vt = randomized_svd(mat, n_components=10, n_iter=10, random_state=42)
    expected_term_embeddings = Vt.T * np.sqrt(Sigma)
    wrong_term_embeddings = U * np.sqrt(Sigma)  # lo que daba el bug

    for i, w in enumerate(vocab):
        assert np.allclose(result[w], expected_term_embeddings[i]), (
            f"El embedding de '{w}' no coincide con Vt.T·sqrt(Sigma) (el cálculo "
            "correcto para embeddings de TÉRMINO). Si coincide con U·sqrt(Sigma) "
            "en su lugar, está devolviendo vectores de UC mal etiquetados como "
            "si fueran de término — el bug U-vs-Vt.T sigue presente."
        )
        assert not np.allclose(result[w], wrong_term_embeddings[i]), (
            f"El embedding de '{w}' coincide con U·sqrt(Sigma) (vectores de UC, "
            "el cálculo incorrecto) — bug U-vs-Vt.T presente."
        )


def test_tiny_matrix_returns_empty_without_crashing(svd_corpus_embeddings):
    """Caso borde ya manejado en el código (ppmi_matrix.shape[0] < 2): debe
    devolver {} en vez de intentar un SVD sobre una matriz degenerada."""
    mat = _random_ppmi(1, 5)
    result = svd_corpus_embeddings(_StubSelf(), mat, ["a", "b", "c", "d", "e"])
    assert result == {}
