"""RAG opción A: índice de ejemplos etiquetados (train, y dev solo al evaluar test).

Troceado de ~300 palabras con 50 de solape (regla del EDA), embeddings multilingües y
recuperación top-k por coseno que excluye la propia noticia.

Limitación conocida: train y dev son México 2018 y no contienen COVID-19; para las noticias de
test sobre la pandemia los vecinos serán de otros temas y aportarán estilo, no hechos.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from llm_fewshot import MAX_WORDS_EJEMPLO, _as_example

EMBEDDER = "BAAI/bge-m3"


def chunk_words(text: str, size: int = 300, overlap: int = 50) -> list:
    """Ventanas de `size` palabras con `overlap` de solape. Un texto corto da un solo trozo."""
    if not 0 <= overlap < size:
        raise ValueError("Se necesita 0 <= overlap < size")
    w = text.split()
    if len(w) <= size:
        return [" ".join(w)] if w else []
    paso, out = size - overlap, []
    for i in range(0, len(w), paso):
        out.append(" ".join(w[i:i + size]))
        if i + size >= len(w):
            break
    return out


# --- Embedders: encode(textos, is_query) -> matriz con filas normalizadas ---

class TfidfEmbedder:
    """Sustituto léxico en CPU (pruebas de humo y recuperador de referencia). Se ajusta con el índice."""
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", min_df=1)
        self.fitted = False

    def encode(self, texts, is_query: bool = False):
        if not self.fitted:
            self.fitted = True
            return self.vec.fit_transform(texts)   # filas ya normalizadas (norma L2)
        return self.vec.transform(texts)


class STEmbedder:
    """Embeddings con sentence-transformers (bge-m3 por defecto). Pensado para Kaggle (GPU)."""

    def __init__(self, name: str = EMBEDDER, batch_size: int = 16, device=None, max_seq_length: int = 512):
        from sentence_transformers import SentenceTransformer
        self.name, self.batch_size = name, batch_size
        self.model = SentenceTransformer(name, device=device)
        self.model.max_seq_length = max_seq_length   # un trozo de 300 palabras cabe de sobra

    def encode(self, texts, is_query: bool = False):
        if "e5" in self.name.lower():   # los modelos E5 esperan estos prefijos
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        return self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)


def _dense(x) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# --- Índice ---

class ExampleIndex:
    """Índice de trozos de noticias etiquetadas. Cada documento se identifica por `split:Id`."""

    def __init__(self, embedder, size: int = 300, overlap: int = 50):
        self.embedder, self.size, self.overlap = embedder, size, overlap
        self.docs = self.chunks = self.emb = None

    def build(self, docs: pd.DataFrame) -> "ExampleIndex":
        """`docs`: Id, label, headline, body, split (salida de `llm.llm_inputs` + split)."""
        self.docs = docs.reset_index(drop=True).copy()
        self.docs["uid"] = self.docs["split"].astype(str) + ":" + self.docs["Id"].astype(str)
        filas = [(i, c) for i, b in enumerate(self.docs["body"]) for c in chunk_words(b, self.size, self.overlap)]
        self.chunks = pd.DataFrame(filas, columns=["doc", "texto"])
        self.emb = self.embedder.encode(self.chunks["texto"].tolist(), is_query=False)
        return self

    def search(self, body: str, k: int = 4, exclude=(), splits=None, per_class: bool = False,
               max_sim: "float | None" = None) -> pd.DataFrame:
        """Los k documentos más parecidos (similitud = máximo entre trozos de consulta y de documento).

        exclude: uids (`split:Id`) que no pueden salir (la propia noticia).
        splits: restringe la reserva (p. ej. ('train',) al evaluar dev).
        per_class=True devuelve k/2 Fake y k/2 Verdadera (los más parecidos de cada clase).
        max_sim: descarta casi duplicados por encima de esa similitud.
        """
        q = self.embedder.encode(chunk_words(body, self.size, self.overlap) or [""], is_query=True)
        sims = _dense(self.emb @ q.T).max(axis=1)                        # por trozo
        d = self.docs.assign(sim=pd.Series(sims).groupby(self.chunks["doc"]).max().reindex(self.docs.index))
        d = d[~d["uid"].isin(set(exclude)) & d["sim"].notna()]
        if splits is not None:
            d = d[d["split"].isin(splits)]
        if max_sim is not None:
            d = d[d["sim"] <= max_sim]
        d = d.sort_values(["sim", "uid"], ascending=[False, True])
        if per_class:
            d = pd.concat([d[d["label"] == y].head(k // 2) for y in (1, 0)]).sort_values("sim", ascending=False)
        return d.head(k).reset_index(drop=True)

    def save(self, path) -> None:
        """Guarda embeddings densos y metadatos (para no recalcular en otra sesión)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, emb=_dense(self.emb).astype(np.float32), doc=self.chunks["doc"].to_numpy(),
                            texto=self.chunks["texto"].to_numpy(dtype=str),
                            docs=self.docs.to_json(orient="records", force_ascii=False),
                            cfg=np.array([self.size, self.overlap]), embedder=getattr(self.embedder, "name", "?"))

    @classmethod
    def load(cls, path, embedder) -> "ExampleIndex":
        z = np.load(path, allow_pickle=False)
        if str(z["embedder"]) != getattr(embedder, "name", "?"):
            raise ValueError(f"El índice se creó con {z['embedder']}, no con {getattr(embedder, 'name', '?')}")
        ix = cls(embedder, int(z["cfg"][0]), int(z["cfg"][1]))
        from io import StringIO
        ix.docs = pd.read_json(StringIO(str(z["docs"])), orient="records", dtype={"Id": int, "label": int})
        ix.chunks = pd.DataFrame({"doc": z["doc"], "texto": z["texto"]})
        ix.emb = z["emb"]
        return ix


def knn_provider(index: ExampleIndex, k: int = 4, query_split: str = "dev", splits=("train",),
                 per_class: bool = True, max_words: int = MAX_WORDS_EJEMPLO):
    """Proveedor few-shot con los k vecinos de cada noticia (para `llm_fewshot.run_examples`).

    Excluye la propia noticia (`query_split:Id`). Orden: del menos al más parecido, de modo que el
    vecino más cercano queda justo antes de la noticia. Con per_class=True hay k/2 de cada clase:
    el modelo ve ejemplos parecidos de ambas etiquetas y no puede limitarse a copiar la mayoritaria.
    """
    def provider(row):
        v = index.search(row.body, k, exclude=[f"{query_split}:{row.Id}"], splits=splits, per_class=per_class)
        v = v.iloc[::-1]
        return [_as_example(r, max_words) for _, r in v.iterrows()], "|".join(v["uid"])

    return provider
