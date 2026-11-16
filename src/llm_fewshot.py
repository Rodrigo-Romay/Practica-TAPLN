"""Few-shot con ejemplos distintos por noticia, contexto (evidencias) y prompts versionados.

Envoltorio sobre `llm.py`, que NO se modifica: se reutilizan `build_messages`, `render`,
`answer_ids` y `score_prompt`. Lo que añade:
  - ejemplos por noticia (un «proveedor» recibe la fila y devuelve sus ejemplos);
  - un bloque de contexto opcional antes de la noticia (evidencias del RAG);
  - `prompt_id`: identificador del prompt que va en el nombre del fichero y en una columna,
    para no mezclar puntuaciones de prompts distintos;
  - registro de tokens, tiempo y ejemplos usados por noticia.
"""
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import llm

MAX_WORDS_EJEMPLO = 150   # mismo recorte que llm.select_examples

SYSTEM_EVIDENCIAS = (
    "Eres un verificador de noticias en español. Lees una noticia y decides si es verdadera o falsa. "
    "Antes de la noticia puede haber evidencias recuperadas automáticamente: úsalas solo si tratan "
    "del mismo hecho; pueden ser irrelevantes. Si no hay evidencias o no sirven, decide con el texto "
    "de la noticia: que no haya evidencias no significa que la noticia sea falsa. "
    "En el texto se han eliminado enlaces y nombres de medios, y las cifras aparecen como NUM. "
    "Responde con una sola palabra: Verdadera o Falsa."
)

PROMPTS = {**llm.PROMPTS, "evidencias": SYSTEM_EVIDENCIAS}


# --- Identificador y nombres de fichero ---

def prompt_id(nombre: str, system: str = llm.PROMPTS["base"], user: str = llm.USER, **config) -> str:
    """`<nombre>-<hash6>`: el hash cubre el prompt de sistema, la plantilla y la configuración.

    Cualquier cambio en el texto del prompt o en cómo se eligen ejemplos/evidencias cambia el id.
    """
    if not re.fullmatch(r"[a-z0-9]+", nombre):
        raise ValueError("El nombre del prompt solo admite minúsculas y dígitos (va en el nombre del fichero)")
    blob = json.dumps([system, user, config], sort_keys=True, ensure_ascii=False, default=str)
    return f"{nombre}-{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:6]}"


def out_name(experimento: str, split: str, clave: str, pid: str) -> str:
    """Nombre del CSV de puntuaciones: <experimento>_<split>_<clave>_<prompt_id>.csv."""
    return f"{experimento}_{split}_{clave}_{pid}.csv"


# --- Proveedores de ejemplos: fila -> ([(titular, cuerpo, label), ...], "ids") ---

def example_pool(inputs: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """Reserva de ejemplos a partir de `llm.llm_inputs` (solo noticias con titular)."""
    pool = inputs[inputs["headline"] != llm.NO_HEADLINE].copy()
    pool["split"] = split
    return pool.reset_index(drop=True)


def _as_example(r, max_words: int = MAX_WORDS_EJEMPLO) -> tuple:
    return (r["headline"], " ".join(str(r["body"]).split()[:max_words]), int(r["label"]))


def fixed_provider(pool: pd.DataFrame, k: int, seed: int = 42, max_words: int = MAX_WORDS_EJEMPLO):
    """Los mismos k ejemplos para todas las noticias (los de `llm.select_examples`)."""
    ejemplos = llm.select_examples(pool, k, seed=seed, max_words=max_words)
    return lambda row: (ejemplos, f"fijos-seed{seed}")


def random_provider(pool: pd.DataFrame, k: int, seed: int = 42, max_words: int = MAX_WORDS_EJEMPLO,
                    query_split: "str | None" = None):
    """k ejemplos al azar distintos para cada noticia (mitad Fake, mitad Verdadera, alternados).

    La semilla depende del Id, así que es reproducible. Si la reserva incluye el conjunto que se
    evalúa (`query_split`), se excluye la propia noticia.
    """
    if k % 2:
        raise ValueError("k debe ser par (mitad Fake, mitad Verdadera)")

    def provider(row):
        cand = pool
        if query_split is not None:
            cand = pool[~((pool["split"] == query_split) & (pool["Id"] == row.Id))]
        rng = np.random.default_rng([seed, int(row.Id)])
        pick = {y: cand[cand["label"] == y].sample(k // 2, random_state=rng) for y in (1, 0)}
        filas = [pick[y].iloc[i] for i in range(k // 2) for y in (1, 0)]
        return [_as_example(r, max_words) for r in filas], "|".join(f"{r['split']}:{r['Id']}" for r in filas)

    return provider


def cached_provider(provider, inputs: pd.DataFrame):
    """Calcula ya los ejemplos de todas las noticias y devuelve un proveedor que solo consulta.

    Sirve para el RAG: se recuperan los vecinos antes de cargar el LLM y se libera el embedder.
    """
    cache = {r.Id: provider(r) for r in inputs.itertuples(index=False)}
    return lambda row: cache[row.Id]


# --- Construcción del prompt ---

def prepare_row(row, provider=None, context=None) -> dict:
    """Ejemplos y contexto de una noticia: {'examples', 'ejemplos', 'context', 'n_evidencias'}."""
    examples, ids = provider(row) if provider else ([], "")
    ctx, n_ev = context(row) if context else ("", 0)
    return {"examples": list(examples), "ejemplos": ids, "context": ctx, "n_evidencias": n_ev}


def render_prompt(tok, headline: str, body: str, examples=None, context: str = "",
                  system: str = llm.PROMPTS["base"], user: str = llm.USER) -> str:
    """Prompt final: sistema + ejemplos (turnos de chat) + [contexto] + noticia."""
    msgs = llm.build_messages(headline, body, examples, system=system, user=user)
    if context:
        msgs[-1]["content"] = f"{context}\n\n{msgs[-1]['content']}"
    return llm.render(tok, msgs)


def _n_tokens(tok, prompt: str) -> int:
    return len(tok(prompt, add_special_tokens=False)["input_ids"])


def fit_prompt(tok, headline, body, prep: dict, system, user, max_tokens: "int | None"):
    """Prompt que cabe en `max_tokens`: si no cabe, quita ejemplos de dos en dos (nunca recorta la noticia)."""
    examples = list(prep["examples"])
    prompt = render_prompt(tok, headline, body, examples, prep["context"], system, user)
    while max_tokens and examples and _n_tokens(tok, prompt) > max_tokens:
        examples = examples[2:]
        prompt = render_prompt(tok, headline, body, examples, prep["context"], system, user)
    return prompt, len(examples)


# --- Puntuación ---

def run_examples(model, tok, inputs: pd.DataFrame, out_path, pid: str, provider=None, context=None,
                 system: str = llm.PROMPTS["base"], user: str = llm.USER, max_tokens: "int | None" = None,
                 save_every: int = 25, log_every: int = 50) -> pd.DataFrame:
    """Como `llm.run`, con ejemplos y contexto por noticia. Guarda en CSV y reanuda si se corta.

    Columnas: las de `llm.run` + prompt_id, n_ejemplos, ejemplos (ids usados) y n_evidencias.
    Si el CSV existente es de otro prompt, se para (no se mezclan puntuaciones).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out_path, keep_default_na=False).to_dict("records") if out_path.exists() else []
    if rows and {str(r.get("prompt_id")) for r in rows} != {pid}:
        raise ValueError(f"{out_path.name} contiene puntuaciones de otro prompt: usa otro fichero")
    done = {r["Id"] for r in rows}
    ids = llm.answer_ids(tok)
    todo = inputs[~inputs["Id"].isin(done)]
    t0 = time.time()
    for k, r in enumerate(todo.itertuples(index=False), 1):
        prep = prepare_row(r, provider, context)
        prompt, n_ej = fit_prompt(tok, r.headline, r.body, prep, system, user, max_tokens)
        t = time.time()
        res = llm.score_prompt(model, tok, prompt, ids)
        if res["nan"] and k == 1:
            raise RuntimeError("Puntuación NaN en la primera noticia: probablemente desbordamiento de fp16. "
                               "Cargar el modelo con dtype=torch.float32.")
        rows.append({"Id": r.Id, "label": r.label, **res, "segundos": time.time() - t, "prompt_id": pid,
                     "n_ejemplos": n_ej, "ejemplos": prep["ejemplos"], "n_evidencias": prep["n_evidencias"]})
        if k % save_every == 0 or k == len(todo):
            pd.DataFrame(rows).to_csv(out_path, index=False)
        if k % log_every == 0:
            print(f"  {k}/{len(todo)}  ({(time.time() - t0) / k:.2f} s/noticia)")
    return pd.DataFrame(rows)


def is_complete(out_path, n: int) -> bool:
    """True si el CSV existe y tiene las n noticias (el experimento se puede saltar)."""
    out_path = Path(out_path)
    return out_path.exists() and len(pd.read_csv(out_path)) == n


def cost_summary(df: pd.DataFrame) -> dict:
    """Coste de una ejecución: tokens y segundos (media y total)."""
    return {"n": len(df), "tokens_media": float(df["n_tokens"].mean()), "tokens_total": int(df["n_tokens"].sum()),
            "s_noticia": float(df["segundos"].mean()), "minutos_total": float(df["segundos"].sum() / 60)}
