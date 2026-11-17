"""RAG opción B: evidencias externas tras una interfaz intercambiable (`EvidenceSource`).

Reglas:
  - se excluye la fuente original de la noticia y su dominio (`Link` solo se usa para esto);
  - si se conoce una fecha límite, se descartan las evidencias posteriores;
  - «sin evidencia» es un caso propio: se le dice al modelo y no equivale a «falsa».
  
"""
import html
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd
import eda_utils as eu

NO_HEADLINE = "no disponible"  

SIN_EVIDENCIA = ("Evidencias recuperadas: ninguna. No se ha encontrado información sobre este hecho; "
                 "eso no indica que la noticia sea falsa ni verdadera.")
CABECERA = "Evidencias recuperadas automáticamente (pueden no tratar del mismo hecho):"


@dataclass
class Evidence:
    titulo: str
    texto: str
    url: str = ""
    fecha: "str | None" = None    
    fuente: str = ""
    score: float = 0.0

    @property
    def dominio(self) -> str:
        return eu.extract_domain(self.url)


class EvidenceSource:
    """Interfaz: devuelve como mucho k evidencias para una consulta, anteriores a `fecha_max` si se da."""
    name = "base"

    def search(self, query: str, k: int = 3, fecha_max: "str | None" = None) -> list:
        raise NotImplementedError


class NullSource(EvidenceSource):
    """Componente simulado: nunca encuentra nada (sirve para medir el efecto del prompt «sin evidencia»)."""
    name = "ninguna"

    def search(self, query, k=3, fecha_max=None):
        return []


class LocalCorpusSource(EvidenceSource):
    """Corpus propio en memoria: DataFrame con titulo, texto y, opcionalmente, url y fecha. TF-IDF."""
    name = "local"

    def __init__(self, corpus: pd.DataFrame, min_score: float = 0.05):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.corpus, self.min_score = corpus.reset_index(drop=True), min_score
        self.vec = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode")
        self.X = self.vec.fit_transform(self.corpus["titulo"] + " " + self.corpus["texto"])

    def search(self, query, k=3, fecha_max=None):
        sims = (self.X @ self.vec.transform([query]).T).toarray().ravel()
        out = []
        for i in sims.argsort()[::-1]:
            if sims[i] < self.min_score or len(out) == k:
                break
            r = self.corpus.iloc[i]
            out.append(Evidence(r["titulo"], r["texto"], r.get("url", "") or "", r.get("fecha") or None,
                                self.name, float(sims[i])))
        return out


class WikipediaSource(EvidenceSource):
    """Wikipedia en español mediante la API de MediaWiki.

    Con `fecha_max` se pide la última revisión anterior a esa fecha (evita leer artículos escritos
    después de la noticia); si la página no existía entonces, se descarta. Sin `fecha_max` se usa el
    resumen actual, que puede describir el hecho a posteriori: úsese solo sabiendo ese riesgo.
    """
    name = "wikipedia"

    def __init__(self, lang: str = "es", pausa: float = 1.0, max_words: int = 150, timeout: int = 15,
                 user_agent: str = "FakeDeS-TAPLN/0.1 (practica universitaria; uso academico; "
                                   "https://github.com/Rodrigo-Romay/Practica-TAPLN)",
                 reintentos: int = 6):
        self.api = f"https://{lang}.wikipedia.org/w/api.php"
        self.base = f"https://{lang}.wikipedia.org/wiki/"
        self.pausa, self.max_words, self.timeout, self.ua = pausa, max_words, timeout, user_agent
        self.reintentos = reintentos

    def _get(self, **params) -> dict:
        """GET a la API con pausa fija y reintentos con espera creciente ante 429/5xx (respeta Retry-After)."""
        import requests
        for intento in range(self.reintentos + 1):
            time.sleep(self.pausa)
            r = requests.get(self.api, params={"format": "json", "formatversion": 2, **params},
                             headers={"User-Agent": self.ua}, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503, 504) and intento < self.reintentos:
                try:
                    espera = float(r.headers.get("Retry-After", ""))
                except ValueError:
                    espera = 0.0
                time.sleep(min(max(espera, 5 * 2 ** intento), 120))   # 5, 10, 20, 40, 80, 120 s
                continue
            r.raise_for_status()
            return r.json()

    def _recorta(self, texto: str) -> str:
        return " ".join(texto.split()[:self.max_words])

    def search(self, query, k=3, fecha_max=None):
        if not query.strip():
            return []
        hits = self._get(action="query", list="search", srsearch=query, srlimit=k).get("query", {}).get("search", [])
        out = []
        for pos, h in enumerate(hits):
            titulo, url = h["title"], self.base + h["title"].replace(" ", "_")
            if fecha_max:
                p = self._get(action="query", prop="revisions", titles=titulo, rvlimit=1, rvdir="older",
                              rvstart=f"{fecha_max}T23:59:59Z", rvprop="timestamp|content", rvslots="main")
                revs = (p.get("query", {}).get("pages") or [{}])[0].get("revisions")
                if not revs:
                    continue   # la página no existía en esa fecha
                texto = strip_wikitext(revs[0]["slots"]["main"]["content"])
                fecha = revs[0]["timestamp"][:10]
            else:
                p = self._get(action="query", prop="extracts", titles=titulo, exintro=1, explaintext=1)
                texto, fecha = (p.get("query", {}).get("pages") or [{}])[0].get("extract", ""), None
            if texto.strip():
                out.append(Evidence(titulo, self._recorta(texto), url, fecha, self.name, 1.0 / (pos + 1)))
        return out


def strip_wikitext(w: str) -> str:
    """Limpieza aproximada de wikitexto: quita plantillas, referencias, ficheros y marcas de enlace."""
    w = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>|<!--.*?-->", " ", w, flags=re.S)
    for _ in range(5):                                 
        w, n = re.subn(r"\{\{[^{}]*\}\}", " ", w)
        if not n:
            break
    w = re.sub(r"\{\|.*?\|\}", " ", w, flags=re.S)    
    w = re.sub(r"\[\[(?:Archivo|File|Imagen|Image|Categoría|Category):[^\]]*\]\]", " ", w, flags=re.I)
    w = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", w)
    w = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", w)
    w = re.sub(r"<[^>]+>|'{2,}|={2,}[^=]+={2,}", " ", w)
    return " ".join(html.unescape(w).split())


# Consulta, filtros y formato 

def make_query(headline: str, body: str, max_words: int = 16) -> str:
    """Consulta a partir del texto: el titular si existe; si no, la primera frase del cuerpo."""
    base = headline if headline and headline != NO_HEADLINE else re.split(r"(?<=[.!?])\s", body, maxsplit=1)[0]
    palabras = [p for p in re.findall(r"\w+", base) if p != "NUM"]
    return " ".join(palabras[:max_words])


def filter_evidence(evs, excluir_dominios=(), fecha_max: "str | None" = None, exigir_fecha: bool = False) -> list:
    """Quita evidencias del dominio original y las posteriores a `fecha_max`.

    Las evidencias sin fecha se conservan salvo que exigir_fecha=True.
    """
    excl = {d for d in excluir_dominios if d}
    out = []
    for e in evs:
        if e.dominio and e.dominio in excl:
            continue
        if fecha_max and e.fecha and e.fecha > fecha_max:
            continue
        if fecha_max and exigir_fecha and not e.fecha:
            continue
        out.append(e)
    return out


def format_evidence(evs, max_words: int = 120, media_re=None) -> str:
    """Bloque de texto para el prompt. Sin evidencias devuelve el aviso explícito `SIN_EVIDENCIA`.

    No se muestran URL ni nombres de medio; el texto pasa por la misma limpieza que la noticia.
    """
    if not evs:
        return SIN_EVIDENCIA
    import preprocessing as pp
    lineas = [CABECERA]
    for i, e in enumerate(evs, 1):
        texto = pp.clean_text(" ".join(e.texto.split()[:max_words]), media_re)
        lineas.append(f"[{i}] {pp.clean_text(e.titulo, media_re)}: {texto}")
    return "\n".join(lineas)


# Recuperación para un conjunto 

def retrieve_all(inputs: pd.DataFrame, links, source: EvidenceSource, path, k: int = 3,
                 fecha_max: "str | None" = None, log_every: int = 50) -> dict:
    """Recupera evidencias para cada noticia y las guarda en JSONL (una línea por noticia).

    `inputs`: Id, headline, body. `links`: URL original de cada noticia, solo para excluir su dominio.
    Si el fichero existe, continúa desde las noticias que faltan. Los fallos de red se guardan como
    error y la noticia queda sin evidencia (se puede repetir borrando esas líneas).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hechos = load_evidence(path) if path.exists() else {}
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for n, (r, link) in enumerate(zip(inputs.itertuples(index=False), links), 1):
            if r.Id in hechos:
                continue
            q, error = make_query(r.headline, r.body), ""
            try:
                evs = filter_evidence(source.search(q, k, fecha_max), [eu.extract_domain(link)], fecha_max)
            except Exception as e:   
                evs, error = [], repr(e)
            seguidos = seguidos + 1 if error else 0
            if seguidos >= 5:
                raise RuntimeError(f"5 errores de red seguidos (último: {error}); se detiene la recuperación. "
                                   "Espera unos minutos y vuelve a ejecutar: continúa donde se quedó.")
            f.write(json.dumps({"Id": int(r.Id), "query": q, "fuente": source.name, "fecha_max": fecha_max,
                                "error": error, "evidencias": [asdict(e) for e in evs]}, ensure_ascii=False) + "\n")
            f.flush()
            hechos[r.Id] = evs
            if n % log_every == 0:
                print(f"  evidencias: {n}/{len(inputs)}")
    return hechos


def load_evidence(path) -> dict:
    """{Id: [Evidence, ...]} desde el JSONL de `retrieve_all`."""
    out = {}
    with Path(path).open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                d = json.loads(linea)
                out[d["Id"]] = [Evidence(**e) for e in d["evidencias"]]
    return out


def evidence_table(path) -> pd.DataFrame:
    """Tabla plana (Id, query, n, error, títulos) para revisar a mano y para la aplicación."""
    filas = []
    with Path(path).open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                d = json.loads(linea)
                filas.append({"Id": d["Id"], "query": d["query"], "n": len(d["evidencias"]), "error": d["error"],
                              "titulos": " | ".join(e["titulo"] for e in d["evidencias"])})
    return pd.DataFrame(filas)


def evidence_context(evidencias: dict, max_words: int = 120, media_re=None):
    """Proveedor de contexto para `llm_fewshot.run_examples`: fila -> (bloque de texto, n evidencias)."""
    def context(row):
        evs = evidencias.get(row.Id, [])
        return format_evidence(evs, max_words, media_re), len(evs)

    return context
