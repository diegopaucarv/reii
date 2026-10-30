"""Tests de la atribución de centinelas y del detector de sign flip entre etapas.

No tocan la BD real: construyen una BD SQLite temporal con el esquema de Optuna
(el mismo de ``data/reii_optuna.db``) y ejercitan ``reii.diag.optuna_db`` por su
``main(argv)``. El módulo vive en el paquete, así que basta el import normal
(pytest añade ``src`` al ``sys.path`` vía ``pyproject.toml``).
"""

from __future__ import annotations

import json
import sqlite3

from reii.diag import optuna_db as D

# Esquema exacto de data/reii_optuna.db (solo las tablas que lee el script).
DDL = """
CREATE TABLE studies (
    study_id INTEGER NOT NULL,
    study_name VARCHAR(512) NOT NULL,
    PRIMARY KEY (study_id)
);
CREATE TABLE study_directions (
    study_direction_id INTEGER NOT NULL,
    direction VARCHAR(8) NOT NULL,
    study_id INTEGER NOT NULL,
    objective INTEGER NOT NULL,
    PRIMARY KEY (study_direction_id),
    UNIQUE (study_id, objective),
    FOREIGN KEY(study_id) REFERENCES studies (study_id)
);
CREATE TABLE trials (
    trial_id INTEGER NOT NULL,
    number INTEGER,
    study_id INTEGER,
    state VARCHAR(8) NOT NULL,
    datetime_start DATETIME,
    datetime_complete DATETIME,
    PRIMARY KEY (trial_id),
    FOREIGN KEY(study_id) REFERENCES studies (study_id)
);
CREATE TABLE trial_user_attributes (
    trial_user_attribute_id INTEGER NOT NULL,
    trial_id INTEGER,
    "key" VARCHAR(512),
    value_json TEXT,
    PRIMARY KEY (trial_user_attribute_id),
    UNIQUE (trial_id, "key"),
    FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
);
CREATE TABLE trial_params (
    param_id INTEGER NOT NULL,
    trial_id INTEGER,
    param_name VARCHAR(512),
    param_value FLOAT,
    distribution_json TEXT,
    PRIMARY KEY (param_id),
    UNIQUE (trial_id, param_name),
    FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
);
CREATE TABLE trial_values (
    trial_value_id INTEGER NOT NULL,
    trial_id INTEGER NOT NULL,
    objective INTEGER NOT NULL,
    value FLOAT,
    value_type VARCHAR(7) NOT NULL,
    PRIMARY KEY (trial_value_id),
    UNIQUE (trial_id, objective),
    FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
);
"""

DIST_JSON = {
    pn: '{"name": "IntDistribution", "attributes": {"log": false, "step": 1, "low": 1, "high": 6}}'
    for pn in ("signflip", "stable")
}

STAGE1 = "synth_stage1_wc_feedface"
STAGE2 = "synth_stage2_select_feedface"

# Rango elegido para que ρ(stable, objetivo monótono) = 0 en ambas etapas.
STABLE = (2, 4, 1, 3)


def _insert_trial(con, tid, sid, values, params, attrs=None):
    con.execute(
        "INSERT INTO trials (trial_id, number, study_id, state) VALUES (?,?,?,?)",
        (tid, tid, sid, "COMPLETE"),
    )
    for obj, v in enumerate(values):
        con.execute(
            "INSERT INTO trial_values (trial_id, objective, value, value_type) "
            "VALUES (?,?,?,?)",
            (tid, obj, float(v), "FINITE"),
        )
    for pn, pv in params.items():
        con.execute(
            "INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json) "
            "VALUES (?,?,?,?)",
            (tid, pn, float(pv), DIST_JSON[pn]),
        )
    for key, val in (attrs or {}).items():
        con.execute(
            'INSERT INTO trial_user_attributes (trial_id, "key", value_json) VALUES (?,?,?)',
            (tid, key, json.dumps(val)),
        )


def make_db(path, only_stage1=False):
    """Dos estudios sintéticos: ``signflip`` invierte ρ en obj0, ``stable`` no.

    Cada estudio lleva un centinela (-2, -1, 0): en etapa 1 con `gate_reason` y
    `k_rejected` (como escribe el código real) y en etapa 2 SIN user attrs
    (BD antigua / atribución no escrita).
    """
    con = sqlite3.connect(path)
    try:
        con.executescript(DDL)
        con.execute(
            "INSERT INTO studies (study_id, study_name) VALUES (1, ?)", (STAGE1,)
        )
        if not only_stage1:
            con.execute(
                "INSERT INTO studies (study_id, study_name) VALUES (2, ?)", (STAGE2,)
            )
        for sid in (1, 2):
            for obj, direction in enumerate(("MAXIMIZE", "MAXIMIZE", "MINIMIZE")):
                con.execute(
                    "INSERT INTO study_directions (study_id, objective, direction) "
                    "VALUES (?,?,?)",
                    (sid, obj, direction),
                )
        # Etapa 1: `signflip` correlaciona POSITIVO con el objetivo 0; `stable`
        # está descorrelacionado (ρ=0) con los tres objetivos.
        for j in range(4):
            _insert_trial(
                con,
                10 + j,
                1,
                (j + 1, j + 1, -(4 - j)),
                {"signflip": j + 1, "stable": STABLE[j]},
            )
        _insert_trial(
            con,
            15,
            1,
            D.SENTINEL,
            {"signflip": 0, "stable": 0},
            {"gate_reason": "K fuera de [2, 20]", "k_rejected": 25, "outcome": "gate"},
        )
        if not only_stage1:
            # Etapa 2: `signflip` invierte el signo en obj0; `stable` sigue a ρ=0.
            for j in range(4):
                _insert_trial(
                    con,
                    20 + j,
                    2,
                    (4 - j, j + 1, -(4 - j)),
                    {"signflip": j + 1, "stable": STABLE[j]},
                )
            _insert_trial(  # sin user attrs a propósito
                con, 25, 2, D.SENTINEL, {"signflip": 0, "stable": 0}
            )
        con.commit()
    finally:
        con.close()
    return path


def _run(tmp_path):
    db = make_db(tmp_path / "diag.db")
    assert D.main(["--db", str(db)]) in (None, 0)
    return db


def test_sign_flip_marks_only_the_inverting_param(tmp_path, capsys):
    _run(tmp_path)
    out = capsys.readouterr().out

    assert "SIGN FLIP ENTRE ETAPAS (posible artefacto de objetivo)" in out
    marked = [ln for ln in out.splitlines() if "<<< INVIERTE" in ln]
    assert len(marked) == 1, marked
    assert "signflip" in marked[0] and "obj[0]" in marked[0]
    assert "stable" not in marked[0]
    # `stable` sí se tabula, pero sin marca.
    assert any("stable" in ln and "obj[0]" in ln for ln in out.splitlines())


def test_gate_reason_breakdown_counts_sentinels(tmp_path, capsys):
    _run(tmp_path)
    out = capsys.readouterr().out

    assert "Centinelas (-2.0, -1.0, 0.0): 1/5 (20%)" in out
    assert "por gate_reason: {'K fuera de [2, 20]': 1}" in out
    assert "k_rejected 25..25 (por exceso del límite superior 20)" in out


def test_sentinel_without_user_attrs_does_not_break(tmp_path, capsys):
    _run(tmp_path)
    out = capsys.readouterr().out

    assert "centinelas sin gate_reason en user attrs: 1" in out


def test_no_pair_means_no_sign_flip_section(tmp_path, capsys):
    db = make_db(tmp_path / "solo_etapa1.db", only_stage1=True)
    assert D.main(["--db", str(db)]) in (None, 0)
    assert "SIGN FLIP ENTRE ETAPAS" not in capsys.readouterr().out


def test_is_sign_flip_threshold():
    assert D.is_sign_flip(0.30, -0.30)
    assert not D.is_sign_flip(0.29, -0.30)
    assert not D.is_sign_flip(0.50, 0.50)
    assert not D.is_sign_flip(None, -0.50)
