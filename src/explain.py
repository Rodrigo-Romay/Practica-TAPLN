"""Explicabilidad: atribución por oclusión de frases (LLM) y términos con más peso (TF-IDF).

Oclusión: se vuelve a puntuar la noticia quitando cada frase y se mide cuánto cambia `logit_diff`.
Es una explicación BASADA EN EL MODELO (qué frases mueven su puntuación), NO una prueba causal ni
una verificación de hechos: las frases interactúan entre sí y quitar una cambia también el contexto.

Este módulo no importa torch: la función de puntuación se le pasa desde fuera
(`llm_score_fn` la construye con un modelo cargado).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

NO_HEADLINE = "no disponible"   # igual que llm.NO_HEADLINE (no se importa llm para no cargar torch)
_FIN_FRASE = re.compile(r"(?<=[.!?…])[\"”»)]*\s+")


def split_sentences(text: str, max_frases: int = 40) -> list:
    """Frases del texto. Si hay más de `max_frases`, se agrupan frases contiguas para acotar el coste."""
    frases = [f.strip() for f in _FIN_FRASE.split(text) if f.strip()]
    if len(frases) <= max_frases:
        return frases
    cortes = np.linspace(0, len(frases), max_frases + 1).round().astype(int)
    return [" ".join(frases[a:b]) for a, b in zip(cortes[:-1], cortes[1:]) if b > a]


def occlusion(score_fn, headline: str, body: str, max_frases: int = 40) -> pd.DataFrame:
    """Atribución por oclusión. `score_fn(titular, cuerpo) -> logit_diff` (mayor = más Fake).

    Columnas: idx (-1 = titular), frase, score_base, score_sin, caida = score_base − score_sin
    (positiva: la frase empuja hacia Fake) e importancia = caida × signo(score_base)
    (positiva: la frase apoya la decisión tomada; la mayor es la frase más determinante).
    Coste: 1 + nº de frases (+1 si hay titular) pasadas por el modelo.
    """
    frases = split_sentences(body, max_frases)
    base = float(score_fn(headline, body))
    signo = 1.0 if base >= 0 else -1.0
    filas = []
    if headline and headline != NO_HEADLINE:
        filas.append((-1, headline, float(score_fn(NO_HEADLINE, body))))
    for i, f in enumerate(frases):
        filas.append((i, f, float(score_fn(headline, " ".join(frases[:i] + frases[i + 1:])))))
    out = pd.DataFrame(filas, columns=["idx", "frase", "score_sin"])
    out.insert(2, "score_base", base)
    out["caida"] = base - out["score_sin"]
    out["importancia"] = out["caida"] * signo
    return out


def occlusion_cost(inputs: pd.DataFrame, s_noticia: float, max_frases: int = 40) -> dict:
    """Coste estimado de la oclusión: pasadas por el modelo y horas (pasadas × s/noticia)."""
    n = sum(1 + len(split_sentences(b, max_frases)) + int(h != NO_HEADLINE)
            for h, b in zip(inputs["headline"], inputs["body"]))
    return {"noticias": len(inputs), "pasadas": int(n), "pasadas_por_noticia": n / max(len(inputs), 1),
            "horas": n * s_noticia / 3600}


def stratified_subset(df: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Subconjunto para precalcular la oclusión, repartido entre aciertos y errores.

    `df`: Id, label, pred (y opcionalmente `covid`). Se reparte a partes iguales entre las cuatro
    celdas de la matriz de confusión (TP, FP, TN, FN), de modo que los errores quedan
    sobrerrepresentados a propósito; dentro de cada celda, proporcional a COVID / no COVID si existe.
    NO es una muestra representativa: sirve para inspeccionar casos, no para estimar métricas.
    """
    d = df.copy()
    d["celda"] = np.select([(d.label == 1) & (d.pred == 1), (d.label == 0) & (d.pred == 1),
                            (d.label == 0) & (d.pred == 0)], ["TP", "FP", "TN"], "FN")
    rng = np.random.default_rng(seed)
    d = d.iloc[rng.permutation(len(d))]
    if "covid" in d.columns:   # intercala COVID / no COVID dentro de cada celda
        d = d.assign(_o=d.groupby(["celda", "covid"]).cumcount()).sort_values("_o", kind="stable").drop(columns="_o")
    elegidas, cupo = [], n // 4
    for _, g in d.groupby("celda"):
        elegidas.append(g.head(cupo))
    sel = pd.concat(elegidas)
    resto = d[~d["Id"].isin(sel["Id"])].head(n - len(sel))   # si una celda es pequeña, se completa
    return pd.concat([sel, resto]).sort_values("Id").reset_index(drop=True)


def run_occlusion(score_fn_for, inputs: pd.DataFrame, out_path, max_frases: int = 40,
                  log_every: int = 10) -> pd.DataFrame:
    """Oclusión de todas las noticias de `inputs`, guardando noticia a noticia (reanudable).

    `score_fn_for(fila) -> score_fn` permite fijar los ejemplos/evidencias de cada noticia.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prev = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    hechas = set(prev["Id"]) if len(prev) else set()
    partes = [prev] if len(prev) else []
    for k, r in enumerate(inputs[~inputs["Id"].isin(hechas)].itertuples(index=False), 1):
        o = occlusion(score_fn_for(r), r.headline, r.body, max_frases)
        o.insert(0, "Id", r.Id)
        partes.append(o)
        pd.concat(partes, ignore_index=True).to_csv(out_path, index=False)
        if k % log_every == 0:
            print(f"  oclusión: {k} noticias")
    return pd.concat(partes, ignore_index=True) if partes else prev


def llm_score_fn(model, tok, e: dict):
    """Fábrica de funciones de puntuación para un experimento de `experiments.experiment`.

    Los ejemplos few-shot y las evidencias se fijan a los de la noticia original; solo cambia el texto.
    """
    import llm
    import llm_fewshot as fs
    ids = llm.answer_ids(tok)

    def for_row(row):
        prep = fs.prepare_row(row, e["provider"], e["context"])

        def score(headline, body):
            prompt, _ = fs.fit_prompt(tok, headline, body, prep, e["system"], llm.USER, e["max_tokens"])
            return llm.score_prompt(model, tok, prompt, ids)["logit_diff"]

        return score

    return for_row


def llm_justification(model, tok, headline: str, body: str, veredicto: str, max_new_tokens: int = 80) -> str:
    """Justificación breve GENERADA por el LLM (opcional).

    Es texto generado a posteriori: puede no reflejar por qué el modelo puntuó así. Se muestra
    separado de la atribución por oclusión y nunca como evidencia.
    """
    import llm
    msgs = [{"role": "system", "content": "Eres un verificador de noticias en español. Sé breve y concreto."},
            {"role": "user", "content": f"Titular: {headline}\n\nCuerpo:\n{body}\n\nLa noticia se ha clasificado "
                                        f"como {veredicto}. Explica en dos frases qué rasgos del texto apoyan esa "
                                        "clasificación. No inventes datos externos."}]
    return llm.generate_answer(model, tok, llm.render(tok, msgs), max_new_tokens=max_new_tokens).strip()


# --- Baseline TF-IDF ---

def tfidf_terms(pipe, text: str, k: int = 8) -> pd.DataFrame:
    """Términos de `text` con más peso en un pipeline TF-IDF + clasificador lineal.

    contribucion = tf-idf del término × coeficiente; positiva empuja hacia Fake, negativa hacia True.
    """
    vec, clf = pipe[0], pipe[-1]
    x = vec.transform([text])
    c = x.multiply(clf.coef_[0]).tocoo()
    nombres = vec.get_feature_names_out()
    d = pd.DataFrame({"termino": nombres[c.col], "contribucion": c.data})
    d = d.reindex(d["contribucion"].abs().sort_values(ascending=False).index)
    out = pd.concat([d[d.contribucion > 0].head(k), d[d.contribucion < 0].head(k)])
    return out.assign(sentido=np.where(out.contribucion > 0, "Fake", "True")).reset_index(drop=True)


def tfidf_terms_table(pipe, ids, texts, k: int = 8) -> pd.DataFrame:
    """`tfidf_terms` para varias noticias (tabla larga con Id)."""
    return pd.concat([tfidf_terms(pipe, t, k).assign(Id=i) for i, t in zip(ids, texts)], ignore_index=True)
