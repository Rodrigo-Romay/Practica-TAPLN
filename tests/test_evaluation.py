"""Tests del evaluador: clase positiva, empates en el rango, métricas frente a scikit-learn y control de test."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import evaluation as ev


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y * 0.8 + rng.normal(0, 1, 300)
    return y, s, (s >= 0.4).astype(int)


def test_fake_es_la_clase_positiva():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([1, 1, 1, 1, 1, 0])   # todas las Fake acertadas, dos True marcadas como Fake
    r = ev.evaluate_full(y, p)
    assert r["recall"] == 1.0 and r["precision"] == pytest.approx(3 / 5)
    assert r["f1_fake"] == pytest.approx(f1_score(y, p, pos_label=1))
    assert r["f1_fake"] != pytest.approx(f1_score(y, p, pos_label=0))
    assert (r["tn"], r["fp"], r["fn"], r["tp"]) == (1, 2, 0, 3)
    # Puntuación alta = Fake: con las puntuaciones invertidas el AUC debe caer por debajo de 0,5
    assert ev.evaluate(y, p, scores=y.astype(float))["auc"] == 1.0
    assert ev.evaluate(y, p, scores=-y.astype(float))["auc"] == 0.0


def test_metricas_coinciden_con_sklearn(datos):
    y, s, p = datos
    r = ev.evaluate_full(y, p, s)
    assert r["f1_fake"] == pytest.approx(f1_score(y, p))
    assert r["precision"] == pytest.approx(precision_score(y, p))
    assert r["recall"] == pytest.approx(recall_score(y, p))
    assert r["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert r["auc"] == pytest.approx(roc_auc_score(y, s))
    assert r["pct_pred_fake"] == pytest.approx(p.mean())


def test_bootstrap_vectorizado_coincide_con_sklearn(datos):
    y, s, p = datos
    s = np.round(s, 1)   # fuerza empates para probar los rangos medios del AUC
    idx = ev._boot_idx(len(y), 20, seed=1)
    m = ev._boot_metrics(y, p, s, idx)
    for b in range(20):
        i = idx[b]
        assert m["f1_fake"][b] == pytest.approx(f1_score(y[i], p[i]))
        assert m["macro_f1"][b] == pytest.approx(f1_score(y[i], p[i], average="macro"))
        assert m["auc"][b] == pytest.approx(roc_auc_score(y[i], s[i]))


def test_intervalo_contiene_la_estimacion(datos):
    y, s, p = datos
    ci = ev.bootstrap_ci(y, p, s, n_boot=500)
    r = ev.evaluate_full(y, p, s)
    for k in ("f1_fake", "auc", "macro_f1"):
        assert ci[k][0] <= r[k] <= ci[k][1]
    assert ev.bootstrap_ci(y, p, s, n_boot=500) == ci   # reproducible con la misma semilla


def test_rango_marca_exactamente_la_fraccion():
    s = np.arange(10, dtype=float)
    assert ev.decide_rank(s, 0.5).sum() == 5
    assert ev.decide_rank(s, 0.4).sum() == 4
    assert list(ev.decide_rank(s, 0.3)) == [0] * 7 + [1] * 3


def test_rango_empates_deterministas_e_independientes_del_orden():
    ids = np.array([10, 11, 12, 13, 14, 15])
    s = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 0.0])    # cuatro empatadas para dos plazas
    p = ev.decide_rank(s, 0.5, ids)
    assert p.sum() == 3
    assert dict(zip(ids, p)) == {10: 1, 11: 1, 12: 1, 13: 0, 14: 0, 15: 0}   # gana el Id menor
    perm = np.array([4, 2, 5, 0, 3, 1])
    p2 = ev.decide_rank(s[perm], 0.5, ids[perm])
    assert dict(zip(ids[perm], p2)) == dict(zip(ids, p))
    # Todas empatadas (puntuaciones saturadas): sigue marcando exactamente la mitad
    assert ev.decide_rank(np.ones(8), 0.5, np.arange(8)).sum() == 4


def test_umbral_por_defecto():
    assert list(ev.decide_threshold([-1.0, 0.0, 2.0])) == [0, 1, 1]
    assert list(ev.decide_threshold([0.2, 0.5, 0.9], 0.5)) == [0, 1, 1]


def test_comparacion_pareada(datos):
    y, s, p = datos
    igual = ev.paired_bootstrap(y, p, p, s, s, n_boot=300)
    assert igual["dif_f1_fake"] == 0 and igual["p_f1_fake"] == 1.0
    peor = 1 - y                                  # sistema que falla siempre
    r = ev.paired_bootstrap(y, y, peor, y.astype(float), -y.astype(float), n_boot=300)
    assert r["dif_f1_fake"] == pytest.approx(1.0) and r["p_f1_fake"] < 0.05 and r["dif_auc"] == pytest.approx(1.0)


def test_macro_f1_no_supera_la_exactitud_en_conjunto_balanceado():
    """Base del argumento del Diario sobre el 0,7666 de ODESIA (no puede ser macro-F1 con acc. 0,7657)."""
    rng = np.random.default_rng(3)
    y = np.array([0, 1] * 143)
    for _ in range(300):
        p = np.where(rng.random(len(y)) < rng.uniform(0.5, 1), y, 1 - y)
        r = ev.evaluate_full(y, p)
        assert r["macro_f1"] <= r["accuracy"] + 1e-12


def test_entrega(tmp_path):
    ev.write_submission([1, 2], [0, 1], tmp_path / "e.txt")
    assert (tmp_path / "e.txt").read_text(encoding="utf-8") == '"fakenews"\t"1"\t"True"\n"fakenews"\t"2"\t"Fake"\n'


def test_lectura_de_predicciones(tmp_path):
    f = tmp_path / "p.csv"
    pd.DataFrame({"Id": [2, 1], "logit_diff": [3.0, -1.0]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        ev.read_predictions(f)
    df = ev.read_predictions(f, score_col="logit_diff", threshold=0.0)
    assert list(df["Id"]) == [1, 2] and list(df["pred"]) == [0, 1]
    pd.DataFrame({"Id": [1, 1], "pred": [0, 1], "score": [0.1, 0.2]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        ev.read_predictions(f)


def test_test_exige_final_y_queda_registrado(tmp_path):
    with pytest.raises(PermissionError):
        ev.load_labels(tmp_path, "test")
    assert ev.logged_test_uses(tmp_path).empty
    ev.log_test_use(tmp_path, "sistema\tx", "yo", "prueba")
    ev.log_test_use(tmp_path, "otro", "yo")
    usos = ev.logged_test_uses(tmp_path)
    assert list(usos["sistema"]) == ["sistema x", "otro"] and list(usos.columns) == ["fecha", "sistema", "quien", "nota"]


def test_umbral_leave_group_out():
    y = np.array([1, 0] * 30)
    s = np.where(y == 1, 3.0, 1.0)             # separables, pero el umbral 0 lo marca todo Fake
    g = np.repeat(["a", "b", "c"], 20)
    t = ev.threshold_leave_group_out(y, s, g, default=0.0)
    assert (t["f1_umbral_resto"] == 1.0).all() and (t["f1_umbral_defecto"] < 1.0).all()
