"""Test fixtures for the REII pipeline."""

from dataclasses import dataclass

import pytest


@dataclass
class MockUCE:
    """Minimal UCE object with .texto attribute — sufficient for pre-merge cache tests."""

    texto: str
    start_char: int = 0
    end_char: int = 0


@pytest.fixture
def sample_uces_por_doc():
    """Provide sample UCEs organized by document for the pre-merge cache test."""
    doc1 = [
        MockUCE("La vida es un recorrido de montañas y valles"),
        MockUCE("Caminar por la vida es como subir una montaña."),
        MockUCE("Las dificultades pueden parecer innecesarias al principio."),
        MockUCE("Aunque no veas la cima, sigue caminando."),
        MockUCE("Al final, te darás cuenta de que superaste montañas."),
    ]
    doc2 = [
        MockUCE("El amor y la vida son los dos grandes temas"),
        MockUCE("Para el ser humano existe el amor."),
        MockUCE("Para el ser humano existe la vida."),
    ]
    doc3 = [
        MockUCE("Hoy es un muy buen día para ir al parque."),
        MockUCE("El parque siempre me hace feliz."),
    ]
    doc4 = [MockUCE("Solo una UCE en este doc")]
    return [doc1, doc2, doc3, doc4]
