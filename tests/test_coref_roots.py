"""Sujetos raíz de COREF: solo las colas de frontera, caché compartida y equivalencia con el flujo anterior."""

import random

import pytest

from reii.seg.root_cache import (
    ROOT_CTX_CHARS,
    RootSubjectCache,
    boundary_tails,
    root_context,
)


class U:  # UCE mínima
    def __init__(self, i, texto):
        self.id, self.texto = f"u{i}", texto


def fake_roots(text):  # determinista; devuelve un iterable
    return {w for w in text.split() if w.istitle()}


class Counter:
    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return fake_roots(text)


def test_root_context_keeps_last_500_chars():
    assert root_context("abc") == "abc"
    long = "x" * 700 + "FIN"
    assert root_context(long) == long[-ROOT_CTX_CHARS:] and root_context(long).endswith(
        "FIN"
    )


def test_boundary_tails_skip_last_group_and_empties():
    a, b, c, d, e, f = (U(i, "t") for i in range(6))
    assert boundary_tails([[a, b], [c], [d, e, f]]) == [b, c]
    assert boundary_tails([[a, b]]) == [] and boundary_tails([]) == []
    assert boundary_tails([[a], [], [b]]) == [a]


def test_prefetch_dedupes_and_second_pass_costs_nothing():
    ex = Counter()
    rc = RootSubjectCache(ex)
    ctx = ["Ana vino", "Luis fue", "Ana vino"]  # repetido
    assert rc.prefetch(ctx) == 2 and len(ex.calls) == 2
    assert rc.prefetch(ctx) == 0 and len(ex.calls) == 2  # pase B2: todo en caché
    assert rc.get("Ana vino") == {"Ana"} and rc.stats()["stanza_calls"] == 2


def test_get_is_lazy_returns_copies_and_does_not_cache_failures():
    state = {"fail": True}

    def flaky(t):
        if state["fail"]:
            raise RuntimeError("boom")
        return {"Ana"}

    rc = RootSubjectCache(flaky)
    assert rc.get("x") == set() and "x" not in rc  # fallo: vacío y sin cachear
    state["fail"] = False
    r = rc.get("x")
    r.add("mutado")
    assert rc.get("x") == {"Ana"}  # la caché no se contamina


def test_batch_extractor_used_and_falls_back_on_error():
    calls = {"batch": 0, "single": 0}

    def single(t):
        calls["single"] += 1
        return fake_roots(t)

    def batch_ok(ts):
        calls["batch"] += 1
        return [fake_roots(t) for t in ts]

    rc = RootSubjectCache(single, batch_ok, batch_size=2)
    rc.prefetch(["Ana a", "Bea b", "Cai c"])
    assert calls["batch"] == 1 and calls["single"] == 1  # lote de 2 + resto individual
    assert rc.get("Cai c") == {"Cai"}

    rc2 = RootSubjectCache(single, lambda ts: 1 / 0, batch_size=4, max_batch_failures=1)
    rc2.prefetch(["Ana a", "Bea b"])
    assert rc2.get("Bea b") == {"Bea"} and rc2._batch is None  # degradó a individual


def test_empty_cache_is_falsy_so_or_pattern_loses_batch():
    """Regresión: ``RootSubjectCache`` define ``__len__``, así que una caché
    VACÍA es falsy. El patrón ``getattr(ps, 'root_cache', None) or
    RootSubjectCache(...)`` crearía una caché nueva SIN el extractor por lotes
    (matando el batching). Hay que comprobar ``is None`` explícito.
    """

    def batch(ts):
        return [fake_roots(t) for t in ts]

    rc = RootSubjectCache(fake_roots, batch, batch_size=2)
    assert rc._batch is not None  # el extractor por lotes SÍ está
    assert not rc  # pero la caché vacía es falsy (len(_store) == 0)
    # el patrón correcto preserva el extractor:
    chosen = rc if rc is not None else RootSubjectCache(fake_roots)
    assert chosen._batch is not None


def test_transient_batch_failure_keeps_batching_for_later_chunks():
    seen = []

    def batch(ts):
        seen.append(list(ts))
        if "Mala x" in ts:
            raise RuntimeError("OOM")
        return [fake_roots(t) for t in ts]

    rc = RootSubjectCache(fake_roots, batch, batch_size=2, max_batch_failures=3)
    rc.prefetch(["Mala x", "Ana a", "Bea b", "Cai c"])
    # el primer lote falla (se resuelve uno a uno) pero el segundo sigue yendo por lote
    assert seen == [["Mala x", "Ana a"], ["Bea b", "Cai c"]] and rc._batch is not None
    assert [rc.get(t) for t in ("Mala x", "Ana a", "Bea b", "Cai c")] == [
        {"Mala"},
        {"Ana"},
        {"Bea"},
        {"Cai"},
    ]

    rc3 = RootSubjectCache(
        fake_roots, lambda ts: 1 / 0, batch_size=2, max_batch_failures=2
    )
    rc3.prefetch(
        ["A a", "B b", "C c", "D d", "E e", "F f"]
    )  # 2 fallos seguidos -> se apaga
    assert rc3._batch is None


def test_progress_is_printed(capsys):
    rc = RootSubjectCache(fake_roots)
    rc.prefetch([f"Nombre{i} x" for i in range(5)], progress_every=2, label="B1 ")
    out = capsys.readouterr().out
    assert "raíces B1 2/5" in out and "raíces B1 4/5" in out and "raíces B1 5/5" in out


def _simulate(groups, decisions, roots_for):
    """Réplica del recorrido voraz: devuelve las raíces consultadas en cada frontera."""
    seen, cur = [], list(groups[0])
    for i in range(1, len(groups)):
        seen.append(roots_for(cur[-1]))
        if decisions[i - 1]:
            cur.extend(groups[i])
        else:
            cur = list(groups[i])
    return seen


@pytest.mark.parametrize("seed", range(25))
def test_new_flow_matches_old_flow_and_never_asks_for_more(seed):
    rng = random.Random(seed)
    n_groups = rng.randint(1, 12)
    groups, k = [], 0
    for _ in range(n_groups):
        size = rng.randint(1, 4)
        groups.append(
            [
                U(k + j, f"Sujeto{k + j} dijo algo {rng.randint(0, 3)}")
                for j in range(size)
            ]
        )
        k += size
    decisions = [rng.random() < 0.5 for _ in range(n_groups)]
    all_uces = [u for g in groups for u in g]

    # flujo anterior: raíces de TODAS las UCEs, indexadas por id
    old = {u.id: set(fake_roots(root_context(u.texto))) for u in all_uces}
    old_seen = _simulate(groups, decisions, lambda u: old[u.id])

    # flujo nuevo: solo colas de frontera, a través de la caché
    ex = Counter()
    rc = RootSubjectCache(ex)
    tails = boundary_tails(groups)
    rc.prefetch([root_context(u.texto) for u in tails])
    new = {u.id: rc.get(root_context(u.texto)) for u in tails}
    new_seen = _simulate(
        groups, decisions, lambda u: new[u.id]
    )  # KeyError si faltara alguna cola

    assert new_seen == old_seen
    assert len(ex.calls) <= len(tails) <= max(0, len(groups) - 1) <= len(all_uces)
