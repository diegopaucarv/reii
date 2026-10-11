import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import diagnose_rf as D  # noqa: E402

FULL = {k: [1] for ks in D.REQUIRED.values() for k in ks}
FULL.update(classes=[0, 1], y_true=[0, 1, 0], cv_balanced_accuracy={"mean": 0.7})


def test_empty_with_flag_off():
    r = D.analyze({}, {"use_rf_shap": False})
    assert r["verdict"] == "SIN_DATOS" and any("use_rf_shap=False" in x for x in r["reasons"])


def test_empty_with_flag_on_and_missing_metadata_columns():
    r = D.analyze({}, {"use_rf_shap": True, "multivariate_metadata": ["grupo"]}, doc_meta_columns=["otra"])
    assert r["verdict"] == "SIN_DATOS"
    assert any("No suitable metadata columns" in x for x in r["reasons"])


def test_unreadable_flag():
    r = D.analyze(None, {})
    assert any("No se pudo leer" in x for x in r["reasons"])


def test_error_saved():
    assert D.analyze({"error": "x"}, {})["verdict"] == "ERROR_GUARDADO"


def test_complete_data_means_reading_problem():
    r = D.analyze(FULL, {"use_rf_shap": True}, stable_cluster_ids=[0, 1, 0], n_stable_uces=3)
    assert r["verdict"] == "OK_DATOS_PRESENTES" and all(r["sections"].values())


def test_stale_classes_and_row_mismatch():
    r = D.analyze(FULL, {"use_rf_shap": True}, stable_cluster_ids=[0, 1, 2], n_stable_uces=10)
    assert r["verdict"] == "DATOS_VIEJOS"


def test_partial_sections():
    d = dict(FULL); d.pop("shap_values")
    r = D.analyze(d, {"use_rf_shap": True}, stable_cluster_ids=[0, 1], n_stable_uces=3)
    assert r["verdict"] == "PARCIAL" and not r["sections"]["SHAP beeswarm"]


def test_json_loader_and_cli(tmp_path, capsys):
    p = tmp_path / "w.json"
    json.dump({"uces": [{"is_stable": True, "cluster_id": 0}], "shap_analysis": {}, "config": {"use_rf_shap": False}}, open(p, "w"))
    assert D.main(["--json", str(p)]) == 1
    assert "SIN_DATOS" in capsys.readouterr().out
