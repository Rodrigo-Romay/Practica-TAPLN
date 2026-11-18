"""Pruebas de humo de few-shot, RAG, explicabilidad y ejecución final con modelos falsos y datos sintéticos."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

import data_utils as du
import evaluation as ev
import experiments as ex
import explain
import llm
import llm_fewshot as fs
import preprocessing as pp
import run_final
from fakes import FakeModel, FakeTok, fake_loader, make_project
from rag import evidence as evd
from rag.index import ExampleIndex, TfidfEmbedder, chunk_words, knn_provider


@pytest.fixture
def proyecto(tmp_path):
    root = make_project(tmp_path)
    splits = du.load_clean_splits(root)
    return root, splits, pp.media_regex_from_splits(splits)


# --- llm.py: firmas en las que se apoya el código nuevo ---

def test_firmas_llm():
    firmas = {
        "llm_inputs": ["df", "media_re", "with_headline", "max_words"],
        "select_examples": ["train_in", "k", "seed", "max_words"],
        "build_messages": ["headline", "body", "examples", "system", "user"],
        "render": ["tok", "messages", "thinking"],
        "load_model": ["name", "quant", "dtype"],
        "release": ["name", "delete_cache"],
        "answer_ids": ["tok"],
        "score_prompt": ["model", "tok", "prompt", "ids"],
        "generate_answer": ["model", "tok", "prompt", "max_new_tokens"],
    }
    for nombre, params in firmas.items():
        assert list(inspect.signature(getattr(llm, nombre)).parameters) == params, nombre
    for const in ("SYSTEM_BASE", "SYSTEM_DETALLADO", "USER", "PROMPTS", "NO_HEADLINE", "ANSWERS", "MAX_WORDS"):
        assert hasattr(llm, const)


# --- Few-shot ---

def test_prompt_id_cambia_con_el_prompt():
    a = fs.prompt_id("fs4base", llm.PROMPTS["base"], llm.USER, k=4)
    assert a == fs.prompt_id("fs4base", llm.PROMPTS["base"], llm.USER, k=4) and a.startswith("fs4base-")
    assert a != fs.prompt_id("fs4base", llm.PROMPTS["base"] + " ", llm.USER, k=4)
    assert a != fs.prompt_id("fs4base", llm.PROMPTS["base"], llm.USER, k=6)
    with pytest.raises(ValueError):
        fs.prompt_id("con_guion")


def test_ejemplos_por_noticia_solo_de_train(proyecto):
    root, splits, media_re = proyecto
    e = ex.experiment(splits, media_re, "dev", modo="fewshot_azar", k=4)
    filas = list(e["inputs"].itertuples(index=False))
    a, ids_a = e["provider"](filas[0])
    b, ids_b = e["provider"](filas[1])
    assert len(a) == 4 and [y for *_, y in a] == [1, 0, 1, 0]
    assert ids_a != ids_b and e["provider"](filas[0])[1] == ids_a          # distintos por noticia, reproducibles
    assert all(u.startswith("train:") for u in (ids_a + "|" + ids_b).split("|"))
    e_test = ex.experiment(splits, media_re, "test", modo="fewshot_azar", k=4)
    usados = "|".join(e_test["provider"](r)[1] for r in e_test["inputs"].itertuples(index=False))
    assert "dev:" in usados and "test:" not in usados
    assert (e_test["inputs"]["label"] == -1).all()                         # las etiquetas de test no viajan
    assert e["pid"] != e_test["pid"]


def test_run_examples_guarda_reanuda_y_no_mezcla_prompts(proyecto, tmp_path):
    root, splits, media_re = proyecto
    tok, model = fake_loader("x")
    e = ex.experiment(splits, media_re, "dev", modo="fewshot_fijo", k=2)
    path = ex.out_path(root, e, "falso")
    assert path.name == f"fewshot_dev_falso_{e['pid']}.csv"
    parcial = dict(e, inputs=e["inputs"].head(10))
    ex.run(model, tok, parcial, path)
    n = model.llamadas
    df = ex.run(model, tok, e, path)
    assert len(df) == len(e["inputs"]) and model.llamadas - n == len(e["inputs"]) - 10   # reanuda
    assert {"prompt_id", "n_ejemplos", "ejemplos", "n_tokens", "segundos", "logit_diff"} <= set(df.columns)
    assert (df["n_ejemplos"] == 2).all() and fs.is_complete(path, len(e["inputs"]))
    otro = ex.experiment(splits, media_re, "dev", modo="fewshot_fijo", k=4)
    with pytest.raises(ValueError):
        ex.run(model, tok, otro, path)
    # El modelo falso separa las clases: el evaluador lo ve
    r = ev.evaluate_predictions(root, ex.to_predictions(df, "falso"), "dev", "falso", n_boot=50)["metricas"]
    assert r["auc"].iloc[0] > 0.7


def test_max_tokens_quita_ejemplos_no_noticia(proyecto):
    root, splits, media_re = proyecto
    tok = FakeTok()
    e = ex.experiment(splits, media_re, "dev", modo="fewshot_fijo", k=4)
    r = next(e["inputs"].itertuples(index=False))
    prep = fs.prepare_row(r, e["provider"], None)
    largo, n4 = fs.fit_prompt(tok, r.headline, r.body, prep, e["system"], llm.USER, None)
    corto, n0 = fs.fit_prompt(tok, r.headline, r.body, prep, e["system"], llm.USER, 150)
    assert n4 == 4 and n0 < 4 and r.body in corto and len(corto) < len(largo)


# --- RAG A ---

def test_troceado():
    t = " ".join(str(i) for i in range(700))
    c = chunk_words(t, 300, 50)
    assert [len(x.split()) for x in c] == [300, 300, 200]
    assert c[1].split()[0] == "250" and c[0].split()[-50:] == c[1].split()[:50]
    assert chunk_words("una dos", 300, 50) == ["una dos"] and chunk_words("", 300, 50) == []


def test_indice_excluye_la_propia_noticia(proyecto):
    root, splits, media_re = proyecto
    pool = ex.pool_for(splits, media_re, "test")
    ix = ExampleIndex(TfidfEmbedder()).build(pool)
    r = pool.iloc[0]
    uid = f"{r['split']}:{r['Id']}"
    assert ix.search(r["body"], 3).iloc[0]["uid"] == uid                    # sin excluir, se encuentra a sí misma
    v = ix.search(r["body"], 4, exclude=[uid])
    assert uid not in set(v["uid"]) and len(v) == 4
    assert set(ix.search(r["body"], 4, splits=("train",))["split"]) == {"train"}
    assert sorted(ix.search(r["body"], 4, per_class=True)["label"]) == [0, 0, 1, 1]
    prov = knn_provider(ix, 4, query_split="train", splits=("train", "dev"))
    ejemplos, ids = prov(next(pool[pool.split == "train"].itertuples(index=False)))
    assert len(ejemplos) == 4 and uid not in ids.split("|")


def test_indice_guardar_y_cargar(proyecto, tmp_path):
    root, splits, media_re = proyecto
    emb = TfidfEmbedder()
    ix = ExampleIndex(emb).build(ex.pool_for(splits, media_re, "dev"))
    ix.save(tmp_path / "i.npz")
    ix2 = ExampleIndex.load(tmp_path / "i.npz", emb)
    cuerpo = ix.docs["body"].iloc[3]
    assert list(ix.search(cuerpo, 3)["uid"]) == list(ix2.search(cuerpo, 3)["uid"])


# --- RAG B ---

def test_evidencias_filtros_y_sin_evidencia():
    evs = [evd.Evidence("A", "texto a", "https://www.medio.com/x", "2018-05-01"),
           evd.Evidence("B", "texto b", "https://otro.org/y", "2020-01-01"),
           evd.Evidence("C", "texto c", "https://otro.org/z", None)]
    assert [e.titulo for e in evd.filter_evidence(evs, ["medio.com"])] == ["B", "C"]        # fuente original fuera
    assert [e.titulo for e in evd.filter_evidence(evs, [], "2018-12-31")] == ["A", "C"]      # posteriores fuera
    assert [e.titulo for e in evd.filter_evidence(evs, [], "2018-12-31", exigir_fecha=True)] == ["A"]
    assert evd.format_evidence([]) == evd.SIN_EVIDENCIA and "falsa ni verdadera" in evd.SIN_EVIDENCIA
    bloque = evd.format_evidence(evs[:1])
    assert "[1] A: texto a" in bloque and "medio.com" not in bloque
    assert evd.make_query("no disponible", "Primera frase con NUM cosas. Segunda.") == "Primera frase con cosas"
    assert evd.NullSource().search("algo") == []
    assert "plantilla" not in evd.strip_wikitext("{{plantilla|x}} Texto [[enlace|visible]]<ref>r</ref>.")


def test_rag_b_de_punta_a_punta(proyecto, tmp_path):
    root, splits, media_re = proyecto
    corpus = pd.DataFrame({"titulo": ["Titular"], "texto": ["titular del comunicado oficial del martes"],
                           "url": ["https://medio1.com/a"], "fecha": ["2018-01-01"]})
    inputs = ex.inputs_for(splits, media_re, "dev")
    path = tmp_path / "ev.jsonl"
    evs = evd.retrieve_all(inputs.head(6), splits["dev"]["Link"].head(6), evd.LocalCorpusSource(corpus), path)
    evs = evd.retrieve_all(inputs, splits["dev"]["Link"], evd.LocalCorpusSource(corpus), path)   # reanuda
    assert len(evs) == len(inputs) == len(evd.evidence_table(path))
    sin = [i for i, v in evs.items() if not v]
    assert sin and len(sin) < len(inputs)        # las noticias de medio1.com se quedan sin evidencia
    e = ex.experiment(splits, media_re, "dev", modo="rag_b", k=3, evidencias=evd.load_evidence(path), fuente="local")
    assert e["system"] == fs.SYSTEM_EVIDENCIAS
    tok, model = fake_loader("x")
    df = ex.run(model, tok, e, tmp_path / "s.csv")
    assert set(df["n_evidencias"]) == {0, 1}
    fila = next(r for r in e["inputs"].itertuples(index=False) if r.Id == sin[0])
    assert e["context"](fila) == (evd.SIN_EVIDENCIA, 0)


# --- Explicabilidad ---

def test_oclusion_encuentra_la_frase_clave(proyecto):
    root, splits, media_re = proyecto
    tok, model = fake_loader("x")
    e = ex.experiment(splits, media_re, "dev", modo="zeroshot")
    fila = next(r for r in e["inputs"].itertuples(index=False) if "bulo" in r.body and r.headline != llm.NO_HEADLINE)
    o = explain.occlusion(explain.llm_score_fn(model, tok, e)(fila), fila.headline, fila.body)
    top = o.sort_values("importancia", ascending=False).iloc[0]
    assert "bulo" in top["frase"] and top["caida"] > 0 and o["idx"].min() == -1
    assert len(o) == len(explain.split_sentences(fila.body)) + 1
    coste = explain.occlusion_cost(e["inputs"], 3.2)
    assert coste["pasadas"] > len(e["inputs"]) and coste["horas"] > 0


def test_oclusion_reanudable_y_subconjunto(tmp_path):
    inputs = pd.DataFrame({"Id": [1, 2, 3], "headline": ["no disponible"] * 3,
                           "body": ["Uno bulo. Dos.", "Tres. Cuatro oficial.", "Cinco. Seis."]})
    llamadas = []

    def fabrica(row):
        def score(h, b):
            llamadas.append(row.Id)
            return float(b.count("bulo") - b.count("oficial"))
        return score

    explain.run_occlusion(fabrica, inputs.head(2), tmp_path / "o.csv")
    n = len(llamadas)
    o = explain.run_occlusion(fabrica, inputs, tmp_path / "o.csv")
    assert set(llamadas[n:]) == {3} and set(o["Id"]) == {1, 2, 3}
    assert o[(o.Id == 2)].sort_values("importancia").iloc[-1]["frase"] == "Cuatro oficial."   # decisión True
    assert len(explain.split_sentences(" ".join(f"Frase {i}." for i in range(100)), 40)) == 40
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Id": range(400), "label": rng.integers(0, 2, 400), "pred": rng.integers(0, 2, 400),
                       "covid": rng.choice(["COVID", "no COVID"], 400)})
    sub = explain.stratified_subset(df, 100)
    assert len(sub) == 100 and sub["Id"].is_unique and (sub["celda"].value_counts() == 25).all()
    pocos = df[(df.label == df.pred) | (df.Id < 10)]
    assert len(explain.stratified_subset(pocos, 100)) == 100


def test_terminos_tfidf(proyecto):
    import baselines as bl
    root, splits, media_re = proyecto
    tr = splits["train"]
    pipe = bl.make_pipe("word", 1.0, "svm").fit(pp.prepare(tr, media_re), tr["label"])
    t = explain.tfidf_terms(pipe, "comunicado bulo bulo del martes oficial")
    assert t[t.termino == "bulo"]["sentido"].iloc[0] == "Fake" and t[t.termino == "oficial"]["sentido"].iloc[0] == "True"


# --- Ejecución final ---

def test_ejecucion_final_una_sola_vez(proyecto, monkeypatch):
    root, splits, media_re = proyecto
    monkeypatch.setitem(ex.MODELOS, "falso-a", {"name": "a"})
    monkeypatch.setitem(ex.MODELOS, "falso-b", {"name": "b"})
    cfg = run_final.default_config()
    cfg["elegido_con"] = "prueba"
    cfg["sistemas"] = [{"clave": "falso-a", "papel": "principal", "modo": "rag_a", "k": 2, "umbral": 0.0},
                       {"clave": "falso-b", "papel": "contraste", "modo": "zeroshot", "umbral": 0.0}]
    run_final.save_config(root, cfg)
    cfg = run_final.load_config(root)
    with pytest.raises(FileNotFoundError):
        run_final.build_predictions(root, cfg)                    # aún no se ha puntuado
    for s in cfg["sistemas"]:
        p = run_final.score_system(root, s, cfg, loader=fake_loader, embedder=TfidfEmbedder())
        assert (pd.read_csv(p)["label"] == -1).all()              # sin etiquetas en las puntuaciones
    todo = run_final.build_predictions(root, cfg)
    assert set(todo["sistema"]) == {"falso-a", "falso-b", "tfidf-char-svm"} and "label" not in todo.columns
    assert ev.logged_test_uses(root).empty                         # hasta aquí no se han tocado las etiquetas
    res = run_final.evaluate_final(root, cfg, "tester", n_boot=50)
    assert len(res["metricas"]) == 6 and len(res["comparacion"]) == 6
    assert {"COVID", "no COVID"} & set(res["desglose"]["grupo"])
    usos = ev.logged_test_uses(root)
    assert len(usos) == 3 and usos["nota"].str.startswith("FINAL").all() and (usos["quien"] == "tester").all()
    d = root / "results" / "final"
    for f in ("predicciones_test.csv", "metricas_test.csv", "comparacion_test.csv", "desglose_test.csv",
              "entrega_falso-a.txt", "entrega_falso-a_rango_transductiva.txt", "manifest.json"):
        assert (d / f).exists(), f
    assert (d / "entrega_falso-a.txt").read_text(encoding="utf-8").startswith('"fakenews"\t"1"\t"')
    with pytest.raises(PermissionError):
        run_final.evaluate_final(root, cfg, "tester", n_boot=50)   # segunda vez: se niega
    run_final.evaluate_final(root, cfg, "tester", repetir="prueba de repetición", n_boot=50)
    assert ev.logged_test_uses(root)["nota"].str.contains("REPETICIÓN").sum() == 3
    assert json.loads((d / "manifest.json").read_text(encoding="utf-8"))["falso-a"]["modo"] == "rag_a"


def test_dev_table_no_admite_test(proyecto):
    with pytest.raises(PermissionError):
        ex.dev_table(proyecto[0], "test")
