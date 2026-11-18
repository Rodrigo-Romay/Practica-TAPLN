"""Ejecución final sobre test: se puntúa y se evalúa UNA SOLA VEZ con la configuración congelada.

La configuración (`results/final/config_final.json`) la escribe `07_calibracion` con decisiones tomadas
solo con dev y leave-topic-out. Pasos (desde la raíz del proyecto):

  python src/run_final.py puntuar                 # GPU (Kaggle): puntúa test con cada sistema; reanudable
  python src/run_final.py baseline                # CPU: TF-IDF entrenado con train + dev
  python src/run_final.py predicciones            # CPU: formato común + predicciones_test.csv
  python src/run_final.py evaluar --final --quien "Nombre"   # CPU: usa las etiquetas; queda registrado

`puntuar`, `baseline` y `predicciones` no leen las etiquetas de test. `evaluar` se niega a repetirse
salvo con --repetir "motivo", que también queda en results/test_evaluations.log.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as bl          # noqa: E402
import data_utils as du         # noqa: E402
import evaluation as ev         # noqa: E402
import preprocessing as pp      # noqa: E402

CONFIG = Path("results") / "final" / "config_final.json"
MARCA = "FINAL"   # prefijo de la nota en el registro de uso de test


def default_config() -> dict:
    """Configuración de partida (zero-shot con los dos finalistas). `07_calibracion` la sustituye."""
    return {
        "fecha": date.today().isoformat(),
        "elegido_con": "PENDIENTE: valores por defecto, sin pasar por 07_calibracion",
        "frac_rango": 0.5,
        "sistemas": [
            {"clave": "gemma4-31b", "papel": "principal", "modo": "zeroshot", "prompt": "base", "k": 4, "umbral": 0.0},
            {"clave": "qwen38-27b", "papel": "contraste", "modo": "zeroshot", "prompt": "base", "k": 4, "umbral": 0.0},
        ],
        "baseline": {"nombre": "tfidf-char-svm", "kind": "char", "clf": "svm", "umbral": 0.0},
        "embedder": "BAAI/bge-m3",
        "evidencias": {"fuente": "wikipedia", "k": 3},
    }


def save_config(root, cfg: dict) -> Path:
    path = Path(root) / CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_config(root) -> dict:
    path = Path(root) / CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Falta {CONFIG}: ejecuta antes 07_calibracion (decisiones con dev)")
    return json.loads(path.read_text(encoding="utf-8"))


def _final_dir(root) -> Path:
    d = Path(root) / "results" / "final"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest(root) -> dict:
    p = _final_dir(root) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _prepare(root):
    splits = du.load_clean_splits(Path(root))
    return splits, pp.media_regex_from_splits(splits)


# --- 1. Puntuar test con los LLM (GPU) ---

def build_experiment(root, s: dict, cfg: dict, splits, media_re, split: str = "test", embedder=None, source=None):
    """Experimento del sistema `s` sobre `split`, con su índice o sus evidencias si los necesita."""
    import experiments as ex
    index = evidencias = None
    fuente = cfg.get("evidencias", {}).get("fuente", "")
    if s["modo"] == "rag_a":
        from rag.index import ExampleIndex, STEmbedder
        embedder = embedder or STEmbedder(cfg["embedder"])
        index = ExampleIndex(embedder).build(ex.pool_for(splits, media_re, split))
    if s["modo"] == "rag_b":
        from rag import evidence as evd
        source = source or {"wikipedia": evd.WikipediaSource, "ninguna": evd.NullSource}[fuente]()
        evidencias = evd.retrieve_all(ex.inputs_for(splits, media_re, split), splits[split]["Link"], source,
                                      Path(root) / "results" / "rag" / f"evidencias_{split}.jsonl",
                                      k=cfg["evidencias"]["k"], fecha_max=ex.FECHA_MAX[split])
    return ex.experiment(splits, media_re, split, modo=s["modo"], prompt=s.get("prompt", "base"), k=s.get("k", 4),
                         index=index, evidencias=evidencias, max_tokens=s.get("max_tokens"), fuente=fuente)


def score_system(root, s: dict, cfg: dict, loader=None, embedder=None, source=None) -> Path:
    """Puntúa test con un sistema y apunta el fichero en el manifiesto. No usa etiquetas. Reanudable."""
    import experiments as ex
    import llm
    import llm_fewshot as fs
    splits, media_re = _prepare(root)
    e = build_experiment(root, s, cfg, splits, media_re, "test", embedder, source)
    path = ex.out_path(root, e, s["clave"], carpeta=_final_dir(root), prefijo="scores")
    if not fs.is_complete(path, len(e["inputs"])):
        m = ex.MODELOS[s["clave"]]
        tok, model = (loader or llm.load_model)(m["name"], m.get("quant"))
        ex.run(model, tok, e, path)
        del model, tok
        llm.release(m["name"], delete_cache=Path("/kaggle").exists())
    man = _manifest(root)
    man[s["clave"]] = {"scores": path.name, "prompt_id": e["pid"], "modo": s["modo"], "umbral": s["umbral"]}
    (_final_dir(root) / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --- 2. Baseline TF-IDF (CPU) ---

def baseline_predictions(root, cfg: dict) -> pd.DataFrame:
    """char-SVM (o el baseline configurado) entrenado con train + dev; predicciones sobre test."""
    b = cfg["baseline"]
    splits, media_re = _prepare(root)
    trdev = pd.concat([splits["train"], splits["dev"]], ignore_index=True)
    m = bl.make_pipe(b["kind"], 1.0, b["clf"]).fit(pp.prepare(trdev, media_re), trdev["label"])
    s = bl.get_scores(m, pp.prepare(splits["test"], media_re))
    out = pd.DataFrame({"Id": splits["test"]["Id"], "sistema": b["nombre"], "score": s,
                        "pred": ev.decide_threshold(s, b["umbral"])}).sort_values("Id").reset_index(drop=True)
    out.to_csv(_final_dir(root) / f"pred_test_{b['nombre']}.csv", index=False)
    return out


# --- 3. Formato común (CPU) ---

def build_predictions(root, cfg: dict) -> pd.DataFrame:
    """pred_test_<sistema>.csv de cada LLM y predicciones_test.csv (todos los sistemas, formato largo)."""
    import experiments as ex
    d, man, n = _final_dir(root), _manifest(root), len(_prepare(root)[0]["test"])
    partes = []
    for s in cfg["sistemas"]:
        if s["clave"] not in man:
            raise FileNotFoundError(f"Falta puntuar {s['clave']} (paso `puntuar`)")
        sc = pd.read_csv(d / man[s["clave"]]["scores"])
        if len(sc) != n:
            raise ValueError(f"{s['clave']}: {len(sc)} de {n} noticias puntuadas; termina el paso `puntuar`")
        if sc["nan"].astype(str).str.lower().eq("true").any():
            raise ValueError(f"{s['clave']}: hay puntuaciones nan")
        p = ex.to_predictions(sc, s["clave"], s["umbral"])
        p.to_csv(d / f"pred_test_{s['clave']}.csv", index=False)
        partes.append(p)
    base = d / f"pred_test_{cfg['baseline']['nombre']}.csv"
    partes.append(pd.read_csv(base) if base.exists() else baseline_predictions(root, cfg))
    todo = pd.concat(partes, ignore_index=True)
    todo["pred_rango"] = 0
    for _, g in todo.groupby("sistema"):   # variante transductiva, guardada aparte de `pred`
        todo.loc[g.index, "pred_rango"] = ev.decide_rank(g["score"], cfg["frac_rango"], g["Id"])
    todo.to_csv(d / "predicciones_test.csv", index=False)
    return todo


# --- 4. Evaluación final (CPU, etiquetas de test, una sola vez) ---

def evaluate_final(root, cfg: dict, quien: str, repetir: "str | None" = None, n_boot: int = 2000) -> dict:
    """Evalúa todos los sistemas en test, los compara y escribe tablas y ficheros de entrega."""
    previas = ev.logged_test_uses(root)
    previas = previas[previas["nota"].astype(str).str.startswith(MARCA)]
    if len(previas) and not repetir:
        raise PermissionError("La evaluación final ya se hizo:\n" + previas.to_string(index=False) +
                              "\nRepetirla exige --repetir \"motivo\" (queda registrado).")
    d, man = _final_dir(root), _manifest(root)
    todo = pd.read_csv(d / "predicciones_test.csv")
    nota = MARCA + (f" (REPETICIÓN: {repetir})" if repetir else "")
    sistemas = {s: g.drop(columns="sistema").reset_index(drop=True) for s, g in todo.groupby("sistema", sort=False)}
    met, des, sen = [], [], []
    for nombre, p in sistemas.items():
        pid = man.get(nombre, {}).get("prompt_id", "tfidf")
        r = ev.evaluate_predictions(root, p, "test", nombre, final=True, quien=quien,
                                    nota=f"{nota}; prompt {pid}; {cfg['elegido_con']}",
                                    n_boot=n_boot, frac_fake=cfg["frac_rango"])
        met.append(r["metricas"]); des.append(r["desglose"]); sen.append(r["sensibilidad"])
        ev.write_submission(p["Id"], p["pred"], d / f"entrega_{nombre}.txt")
    comp = ev.compare_predictions(root, sistemas, "test", final=True, n_boot=n_boot, frac_fake=cfg["frac_rango"])
    out = {"metricas": pd.concat(met, ignore_index=True), "desglose": pd.concat(des, ignore_index=True),
           "sensibilidad": pd.concat(sen, ignore_index=True), "comparacion": comp}
    for k, t in out.items():
        t.to_csv(d / f"{k}_test.csv", index=False)
    principal = next((s["clave"] for s in cfg["sistemas"] if s.get("papel") == "principal"), cfg["sistemas"][0]["clave"])
    p = sistemas[principal]
    ev.write_submission(p["Id"], p["pred_rango"], d / f"entrega_{principal}_rango_transductiva.txt")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ejecución final sobre test (una sola vez).")
    ap.add_argument("paso", choices=["puntuar", "baseline", "predicciones", "evaluar"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--solo", default=None, help="puntuar solo este sistema (clave)")
    ap.add_argument("--final", action="store_true", help="obligatorio para `evaluar`")
    ap.add_argument("--quien", default=None, help="quién lanza la evaluación final")
    ap.add_argument("--repetir", default=None, help="motivo para repetir la evaluación final")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else du.project_root(Path(__file__).resolve().parent)
    cfg = load_config(root)
    if a.paso == "puntuar":
        for s in cfg["sistemas"]:
            if a.solo in (None, s["clave"]):
                print("Puntuado:", score_system(root, s, cfg))
    elif a.paso == "baseline":
        print(baseline_predictions(root, cfg).groupby("sistema")["pred"].mean().rename("% predicho Fake"))
    elif a.paso == "predicciones":
        print(build_predictions(root, cfg).groupby("sistema")[["pred", "pred_rango"]].mean())
    else:
        if not a.final or not a.quien:
            ap.error("`evaluar` exige --final y --quien \"Nombre\"")
        res = evaluate_final(root, cfg, a.quien, a.repetir, a.boot)
        cols = ["modelo", "decision", "f1_fake", "f1_fake_inf", "f1_fake_sup", "auc", "macro_f1", "pct_pred_fake"]
        print(res["metricas"][cols].round(3).to_string(index=False))
        print(res["comparacion"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
