"""Clasificación zero/few-shot con LLMs abiertos (pensado para Kaggle, GPU T4/P100).

No se genera texto: se hace una sola pasada por el modelo y se lee la probabilidad del
primer token de la respuesta («Falsa» frente a «Verdadera»). Así cada noticia tiene una
puntuación continua (p_fake) y se pueden calcular AUC y la decisión por rango, igual que
con los baselines TF-IDF.
"""
import gc
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import preprocessing as pp

MAX_WORDS = 800   # regla del EDA: texto íntegro hasta ~800 palabras

SYSTEM = (
    "Eres un verificador de noticias en español. Lees una noticia y decides si es verdadera o falsa. "
    "En el texto se han eliminado enlaces y nombres de medios, y las cifras aparecen como NUM. "
    "Responde con una sola palabra: Verdadera o Falsa."
)

USER = (
    "Titular: {headline}\n\n"
    "Cuerpo:\n{body}\n\n"
    "¿Esta noticia es verdadera o falsa? Responde solo «Verdadera» o «Falsa»."
)

# Variante del prompt de sistema para el ajuste (4.2): da criterios de verificación, pero sin
# atajos de superficie (mayúsculas, longitud, cifras), que el EDA descartó como señal.
SYSTEM_DETALLADO = (
    "Eres un verificador de noticias en español. Lees una noticia y decides si es verdadera o falsa. "
    "Fíjate en si los hechos son concretos y verosímiles, si las afirmaciones se atribuyen a fuentes "
    "identificables, si el relato es coherente y si hay exageración, alarmismo o afirmaciones "
    "extraordinarias sin respaldo. "
    "En el texto se han eliminado enlaces y nombres de medios, y las cifras aparecen como NUM. "
    "Responde con una sola palabra: Verdadera o Falsa."
)

PROMPTS = {"base": SYSTEM, "detallado": SYSTEM_DETALLADO}

NO_HEADLINE = "no disponible"

# Variantes de la respuesta; de cada una se toma el primer token.
ANSWERS = {
    "fake": ["Falsa", "falsa", "FALSA", "Falso", "falso"],
    "true": ["Verdadera", "verdadera", "VERDADERA", "Verdadero", "verdadero"],
}


# --- Entradas ---

def llm_inputs(df: pd.DataFrame, media_re, with_headline: bool = True,
               max_words: int = MAX_WORDS) -> pd.DataFrame:
    """Titular y cuerpo limpios por separado (el prompt los muestra en campos distintos).

    Misma limpieza que `preprocessing.prepare`; el cuerpo se corta a `max_words` palabras.
    """
    heads, bodies = [], []
    for h, t in zip(df["Headline"], df["Text"]):
        body = pp.clean_text(pp.body_text(h, t), media_re)
        bodies.append(" ".join(body.split()[:max_words]))
        head = pp.clean_text(h, media_re) if (with_headline and h) else ""
        heads.append(head or NO_HEADLINE)
    return pd.DataFrame({"Id": df["Id"].to_numpy(), "label": df["label"].to_numpy(),
                         "headline": heads, "body": bodies})


def select_examples(train_in: pd.DataFrame, k: int, seed: int = 42, max_words: int = 150) -> list:
    """k ejemplos few-shot de train, mitad Fake y mitad Verdadera, alternados (Fake, True, Fake...).

    Se eligen al azar con semilla fija entre las noticias con titular y se acortan a `max_words`
    palabras para que el prompt no se dispare. Solo sale de train: nunca de dev ni de test.
    """
    if k % 2:
        raise ValueError("k debe ser par (mitad Fake, mitad Verdadera)")
    pool = train_in[train_in["headline"] != NO_HEADLINE]
    pick = {y: pool[pool["label"] == y].sample(k // 2, random_state=seed) for y in (1, 0)}
    out = []
    for i in range(k // 2):
        for y in (1, 0):
            r = pick[y].iloc[i]
            out.append((r["headline"], " ".join(r["body"].split()[:max_words]), int(y)))
    return out


def build_messages(headline: str, body: str, examples=None,
                   system: str = SYSTEM, user: str = USER) -> list:
    """Mensajes de chat. `examples` = [(titular, cuerpo, label), ...] para few-shot."""
    msgs = [{"role": "system", "content": system}]
    for h, b, y in examples or []:
        msgs.append({"role": "user", "content": user.format(headline=h, body=b)})
        msgs.append({"role": "assistant", "content": "Falsa" if y == 1 else "Verdadera"})
    msgs.append({"role": "user", "content": user.format(headline=headline, body=body)})
    return msgs


def render(tok, messages: list, thinking: bool = False) -> str:
    """Aplica la plantilla de chat del modelo.

    `enable_thinking` solo lo usa Qwen3 (las demás plantillas lo ignoran). Si la plantilla
    no admite el rol system, se antepone al primer mensaje de usuario.
    """
    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    try:
        return tok.apply_chat_template(messages, **kw)
    except Exception:
        sys_msg, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
        rest[0]["content"] = f"{sys_msg}\n\n{rest[0]['content']}"
        return tok.apply_chat_template(rest, **kw)


# --- Modelo ---

def _default_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    # bf16 solo en GPUs Ampere o posteriores; T4 y P100 van en fp16.
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load_model(name: str, quant: str | None = None, dtype=None):
    """Carga tokenizador y modelo repartido entre las GPUs disponibles.

    quant=None (fp16/bf16) o '4bit' (bitsandbytes NF4).
    """
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype or _default_dtype()
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    kw = {dtype_key: dtype, "device_map": "auto" if torch.cuda.is_available() else None}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except ValueError:
        # Gemma 3 (4B+) es multimodal y puede no estar registrado como CausalLM.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    model.eval()
    return tok, model


def release(name: str | None = None, delete_cache: bool = False) -> None:
    """Libera la GPU y, si se pide, borra el modelo de la caché de Hugging Face (disco de Kaggle).

    Antes hay que hacer `del model, tok` en el notebook: aquí no se puede soltar esa referencia.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if delete_cache and name:
        d = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{name.replace('/', '--')}"
        shutil.rmtree(d, ignore_errors=True)


def answer_ids(tok) -> dict:
    """Ids del primer token de cada variante de respuesta; se quitan los que comparten ambas clases."""
    ids = {}
    for cls, words in ANSWERS.items():
        s = set()
        for w in words:
            for v in (w, " " + w):
                enc = tok.encode(v, add_special_tokens=False)
                if enc:
                    s.add(enc[0])
        ids[cls] = s
    both = ids["fake"] & ids["true"]
    out = {k: sorted(v - both) for k, v in ids.items()}
    if not out["fake"] or not out["true"]:
        raise ValueError(f"No se pudieron separar los tokens de respuesta: {out}")
    return out


@torch.inference_mode()
def next_token_logprobs(model, tok, prompt: str) -> tuple[torch.Tensor, int]:
    """Log-probabilidades del siguiente token tras el prompt (una pasada, sin generar)."""
    # La plantilla de chat ya incluye los tokens especiales (BOS, etc.).
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    try:
        logits = model(**enc, logits_to_keep=1).logits
    except TypeError:
        logits = model(**enc).logits
    return torch.log_softmax(logits[0, -1].float(), dim=-1), enc["input_ids"].shape[1]


def score_prompt(model, tok, prompt: str, ids: dict) -> dict:
    """p_fake = P(Falsa) / (P(Falsa) + P(Verdadera)) y la masa total en esas respuestas."""
    lp, n_tok = next_token_logprobs(model, tok, prompt)
    lf = torch.logsumexp(lp[ids["fake"]], 0).item()
    lt = torch.logsumexp(lp[ids["true"]], 0).item()
    return {
        "logit_diff": lf - lt,
        "p_fake": float(1 / (1 + np.exp(-(lf - lt)))),
        "p_formato": float(np.exp(lf) + np.exp(lt)),   # ~1 si el modelo responde en el formato pedido
        "top_token": tok.decode([int(lp.argmax())]),
        "n_tokens": n_tok,
        "nan": bool(not np.isfinite(lf) or not np.isfinite(lt)),
    }


def run(model, tok, inputs: pd.DataFrame, out_path, thinking: bool = False,
        examples=None, system: str = SYSTEM, save_every: int = 25, log_every: int = 50) -> pd.DataFrame:
    """Puntúa todas las filas de `inputs` y guarda en CSV; si se corta, reanuda desde lo guardado.

    `thinking=True` no se admite: con el modo thinking el primer token es el inicio del
    razonamiento, no la respuesta, y p_fake no tendría sentido.
    """
    if thinking:
        raise ValueError("thinking no es compatible con puntuar el primer token de la respuesta")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    done = {r["Id"] for r in rows}
    ids = answer_ids(tok)
    todo = inputs[~inputs["Id"].isin(done)]
    t0 = time.time()
    for k, r in enumerate(todo.itertuples(index=False), 1):
        prompt = render(tok, build_messages(r.headline, r.body, examples, system=system), thinking)
        t = time.time()
        rows.append({"Id": r.Id, "label": r.label, **score_prompt(model, tok, prompt, ids),
                     "segundos": time.time() - t})
        if k % save_every == 0 or k == len(todo):
            pd.DataFrame(rows).to_csv(out_path, index=False)
        if k % log_every == 0:
            print(f"  {k}/{len(todo)}  ({(time.time() - t0) / k:.2f} s/noticia)")
    return pd.DataFrame(rows)


@torch.inference_mode()
def generate_answer(model, tok, prompt: str, max_new_tokens: int = 8) -> str:
    """Respuesta generada (solo para comprobar a mano que el formato es el esperado)."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
