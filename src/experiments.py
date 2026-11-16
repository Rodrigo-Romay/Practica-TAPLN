"""Definición común de los experimentos con LLM (notebooks 04–06, ejecución final y explicabilidad).

Un experimento = conjunto a puntuar + modo (zero-shot, few-shot, RAG) + prompt. Los notebooks de dev
y la ejecución final llaman a `experiment`, así que test se puntúa exactamente con el código probado en dev.

Regla de los ejemplos: al puntuar dev (o train) la reserva es SOLO train; al puntuar test, train + dev.
"""
from pathlib import Path

import pandas as pd

import llm
import llm_fewshot as fs

# Mismas claves que en notebooks/03_llm.ipynb
MODELOS = {
    "qwen3-8b":   {"name": "Qwen/Qwen3-8B"},
    "gemma3-12b": {"name": "google/gemma-3-12b-it", "quant": "4bit"},
    "qwen38-27b": {"name": "Qwen/Qwen3.8-27B", "quant": "4bit"},
    "gemma4-31b": {"name": "google/gemma-4-31B-it", "quant": "4bit"},
}
FINALISTAS = ("gemma4-31b", "qwen38-27b")

MODOS = {"zeroshot": "zs", "fewshot_fijo": "fs", "fewshot_azar": "fa", "rag_a": "ra", "rag_b": "rb"}
EXPERIMENTO = {"zeroshot": "zeroshot", "fewshot_fijo": "fewshot", "fewshot_azar": "fewshot",
               "rag_a": "ragA", "rag_b": "ragB"}
FECHA_MAX = {"train": "2018-12-31", "dev": "2018-12-31", "test": "2021-03-01"}   # cotas para las evidencias


def inputs_for(splits: dict, media_re, split: str) -> pd.DataFrame:
    """Entradas del LLM. En test la etiqueta se sustituye por -1: no viaja a los ficheros de puntuación."""
    df = llm.llm_inputs(splits[split], media_re)
    if split == "test":
        df["label"] = -1
    return df


def pool_for(splits: dict, media_re, split: str) -> pd.DataFrame:
    """Reserva de ejemplos etiquetados: train (para dev) o train + dev (para test). Nunca test."""
    usar = ("train", "dev") if split == "test" else ("train",)
    return pd.concat([fs.example_pool(llm.llm_inputs(splits[s], media_re), s) for s in usar], ignore_index=True)


def experiment(splits: dict, media_re, split: str, modo: str = "zeroshot", prompt: str = "base", k: int = 4,
               seed: int = 42, index=None, evidencias: "dict | None" = None, per_class: bool = True,
               max_tokens: "int | None" = None, fuente: str = "") -> dict:
    """Devuelve lo necesario para `run`: inputs, provider, context, system, prompt_id y nombre.

    modo: zeroshot | fewshot_fijo | fewshot_azar | rag_a (necesita `index`) | rag_b (necesita `evidencias`).
    prompt: clave de `llm_fewshot.PROMPTS` (base, detallado, evidencias).
    """
    if modo not in MODOS:
        raise ValueError(f"modo desconocido: {modo}")
    if modo == "rag_b" and prompt == "base":
        prompt = "evidencias"
    system = fs.PROMPTS[prompt]
    inputs, provider, context = inputs_for(splits, media_re, split), None, None
    reserva = ("train", "dev") if split == "test" else ("train",)
    cfg = {"modo": modo, "reserva": reserva}
    if modo == "zeroshot":
        nombre = f"zs{prompt}"
    elif modo in ("fewshot_fijo", "fewshot_azar"):
        pool = pool_for(splits, media_re, split)
        provider = (fs.fixed_provider(pool, k, seed) if modo == "fewshot_fijo"
                    else fs.random_provider(pool, k, seed))
        nombre, cfg = f"{MODOS[modo]}{k}{prompt}", {**cfg, "k": k, "seed": seed}
    elif modo == "rag_a":
        if index is None:
            raise ValueError("rag_a necesita un ExampleIndex")
        from rag.index import knn_provider
        # Los vecinos se calculan ahora: después se puede liberar el embedder antes de cargar el LLM
        provider = fs.cached_provider(knn_provider(index, k, query_split=split, splits=reserva,
                                                   per_class=per_class), inputs)
        nombre = f"ra{k}{prompt}"
        cfg = {**cfg, "k": k, "per_class": per_class, "embedder": getattr(index.embedder, "name", "?"),
               "chunk": [index.size, index.overlap]}
    else:
        if evidencias is None:
            raise ValueError("rag_b necesita las evidencias recuperadas (rag.evidence.retrieve_all)")
        from rag.evidence import CABECERA, SIN_EVIDENCIA, evidence_context
        context = evidence_context(evidencias, media_re=media_re)
        nombre = f"rb{k}{prompt}"
        cfg = {**cfg, "k": k, "fuente": fuente, "fecha_max": FECHA_MAX[split], "textos": [CABECERA, SIN_EVIDENCIA]}
    return {"split": split, "modo": modo, "inputs": inputs, "provider": provider, "context": context,
            "system": system, "max_tokens": max_tokens, "experimento": EXPERIMENTO[modo],
            "pid": fs.prompt_id(nombre, system, llm.USER, **cfg)}


def out_path(root, e: dict, clave: str, carpeta: "str | None" = None, prefijo: "str | None" = None) -> Path:
    """results/llm/<experimento>_<split>_<clave>_<prompt_id>.csv (o la carpeta/prefijo que se indique)."""
    carpeta = Path(carpeta) if carpeta else Path(root) / "results" / "llm"
    return carpeta / fs.out_name(prefijo or e["experimento"], e["split"], clave, e["pid"])


def run(model, tok, e: dict, path) -> pd.DataFrame:
    """Puntúa el experimento (reanudable)."""
    return fs.run_examples(model, tok, e["inputs"], path, e["pid"], provider=e["provider"],
                           context=e["context"], system=e["system"], max_tokens=e["max_tokens"])


def run_all(root, experimentos: list, claves=FINALISTAS, borrar_cache: bool = False, loader=None) -> list:
    """Ejecuta varios experimentos con varios modelos: un modelo cada vez, saltando lo ya completo.

    Tras cada modelo libera la GPU y, con borrar_cache=True (Kaggle), borra el modelo del disco.
    """
    import shutil
    hechos = []
    for clave in claves:
        pend = [e for e in experimentos if not fs.is_complete(out_path(root, e, clave), len(e["inputs"]))]
        hechos += [out_path(root, e, clave) for e in experimentos]
        if not pend:
            print(f"{clave}: ya hecho")
            continue
        m = MODELOS[clave]
        print(f"== {clave} ({m['name']}): {len(pend)} experimento(s)")
        tok, model = (loader or llm.load_model)(m["name"], m.get("quant"))
        for e in pend:
            print(f"-- {e['experimento']} {e['pid']}")
            print("  ", fs.cost_summary(run(model, tok, e, out_path(root, e, clave))))
        del model, tok
        llm.release(m["name"], delete_cache=borrar_cache)
        if borrar_cache:
            shutil.rmtree("/tmp/hf/hub", ignore_errors=True)
    return hechos


def show_prompt(e: dict, i: int = 0, recorte: int = 300) -> str:
    """Texto del prompt de la noticia i (sin plantilla de chat), con los cuerpos recortados para leerlo."""
    r = list(e["inputs"].itertuples(index=False))[i]
    prep = fs.prepare_row(r, e["provider"], e["context"])
    corta = lambda t: t if len(t) <= recorte else t[:recorte] + " [...]"
    msgs = llm.build_messages(r.headline, corta(r.body), [(h, corta(b), y) for h, b, y in prep["examples"]],
                              system=e["system"])
    if prep["context"]:
        msgs[-1]["content"] = f"{prep['context']}\n\n{msgs[-1]['content']}"
    return "\n".join(f"[{m['role']}] {m['content']}\n" for m in msgs)


def to_predictions(scores: pd.DataFrame, sistema: str, umbral: float = 0.0) -> pd.DataFrame:
    """CSV de puntuaciones de LLM -> formato común del evaluador (Id, sistema, score, pred, ...)."""
    import evaluation as ev
    out = pd.DataFrame({"Id": scores["Id"], "sistema": sistema, "score": scores["logit_diff"],
                        "pred": ev.decide_threshold(scores["logit_diff"], umbral)})
    for c in ("p_fake", "prompt_id", "n_tokens", "segundos"):
        if c in scores.columns:
            out[c] = scores[c].to_numpy()
    return out.sort_values("Id").reset_index(drop=True)


def dev_table(root, split: str = "dev", n_boot: int = 2000) -> pd.DataFrame:
    """Tabla comparativa de todos los CSV de puntuaciones de un conjunto de train/dev en results/llm/.

    Una fila por fichero y decisión (umbral 0 y rango 50 %, transductiva), con IC de F1 y AUC.
    """
    import evaluation as ev
    if split == "test":
        raise PermissionError("Test solo se evalúa con run_final.py --final")
    filas = []
    for p in sorted((Path(root) / "results" / "llm").glob(f"*_{split}_*.csv")):
        if p.name.endswith("_resumen.csv"):
            continue
        r = pd.read_csv(p)
        if "logit_diff" not in r.columns:
            continue
        res = ev.evaluate_predictions(root, to_predictions(r, p.stem), split, p.stem, n_boot=n_boot)["metricas"]
        filas.append(res.assign(s_noticia=r["segundos"].mean(), tokens=r["n_tokens"].mean(), completo=len(r)))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
