# Utilidades reutilizables del EDA de FakeDeS
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

SEED = 42
SPLITS = ("train", "dev", "test")


def project_root(start=None) -> Path:
    """Sube desde `start` (por defecto, el cwd) hasta la carpeta que contiene data/ y src/."""
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "data").is_dir() and (cand / "src").is_dir():
            return cand
    raise FileNotFoundError("No se encuentra la raíz del proyecto (data/ y src/)")


def load_splits(root: Path) -> dict:
    """Lee data/processed/{train,dev,test}.csv conservando las cadenas vacías (no NaN)."""
    out = {}
    for s in SPLITS:
        df = pd.read_csv(root / "data" / "processed" / f"{s}.csv", keep_default_na=False)
        df["split"] = s
        out[s] = df
    return out



def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def body_text(headline: str, text: str) -> str:
    """Cuerpo sin el titular repetido."""
    h, t = _norm_ws(headline), _norm_ws(text)
    if h and t.startswith(h):
        t = t[len(h):].lstrip()
    return t


def model_input(headline: str, text: str) -> str:
    """Titular + cuerpo sin duplicar (si no hay titular, solo el cuerpo)."""
    b = body_text(headline, text)
    h = _norm_ws(headline)
    return f"{h}\n\n{b}" if h else b


def add_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = [body_text(h, t) for h, t in zip(df["Headline"], df["Text"])]
    df["input"] = [model_input(h, t) for h, t in zip(df["Headline"], df["Text"])]
    df["domain"] = df["Link"].map(extract_domain)
    return df

# Dominios

_SLD = {"com", "co", "org", "gob", "gov", "net", "edu", "gub"}


def extract_domain(url: str) -> str:
    """Dominio registrable de una URL, sin 'www.' ni subdominios."""
    url = (url or "").strip()
    if not url:
        return ""
    host = urlparse(url if "//" in url else "//" + url).netloc.lower().split(":")[0]
    labels = [l for l in host.split(".") if l]
    if len(labels) >= 3 and labels[-2] in _SLD and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


# Limpieza

URL_RE = re.compile(r"(?:https?://|www\.)\S+|\*URL\*|pic\.twitter\.com/\S+", re.I)
NUM_RE = re.compile(r"\*NUMBER\*|\d+(?:[.,]\d+)*")

# Patrones de posible fuga léxica.
LEAK_PATTERNS = {
    "url": URL_RE,
    "credito_foto": re.compile(
        r"\b(?:foto|fotograf[ií]a|imagen|cr[eé]ditos?|ilustraci[oó]n|captura(?: de pantalla)?)\s*:\s*[^.\n]{0,60}", re.I),
    "fuente": re.compile(r"\bfuente\s*:\s*[^.\n]{0,60}", re.I),
    "redaccion": re.compile(r"\bredacci[oó]n\b(?:\s+[A-ZÁÉÍÓÚ]\w+)?", re.I),
    "con_informacion_de": re.compile(r"\b(?:con|de)\s+informaci[oó]n\s+de\s+[^.\n]{0,60}", re.I),
    "via": re.compile(r"\bv[ií]a\s*:\s*@?\w+|\bv[ií]a\s+@\w+", re.I),
    "lee_tambien": re.compile(
        r"\b(?:lee|leer)\s+(?:tambi[eé]n|m[aá]s)\s*:?[^.\n]{0,80}|te puede interesar|s[ií]guenos en\S*|suscr[ií]bete\S*", re.I),
    "dateline": re.compile(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?: [A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?\s*\.\s?[-–—]+\s"),
    "agencia": re.compile(r"\b(?:AFP|EFE|AP|Notimex|Reuters|Europa Press|Associated Press)\b"),
    "mencion_hashtag": re.compile(r"(?<!\w)[@#]\w+"),
}

# Nombres de medio que no se pueden tratar como medio porque son palabras corrientes.
_GENERIC_NAMES = {"proceso", "retroceso", "reforma", "milenio", "censura 0"}

MEDIA_EXTRA = [
    "El País", "El Mundo", "La Vanguardia", "Infobae", "Newtral", "Maldita", "AFP Factual", "AFPFactual",
    "El Universal", "Excélsior", "La Jornada", "Animal Político", "Aristegui Noticias", "El Financiero",
    "El Economista", "BBC Mundo", "BBC News", "CNN", "Telemundo", "Univision", "Agencia EFE", "Huffpost",
]

_ACCENT = {"a": "[aáàä]", "e": "[eéèë]", "i": "[iíìï]", "o": "[oóòö]", "u": "[uúùü]", "n": "[nñ]"}


def _accent_insensitive(name: str) -> str:
    return r"\s+".join("".join(_ACCENT.get(c.lower(), re.escape(c)) for c in w) for w in name.split())


def build_media_regex(names) -> re.Pattern:
    """Regex (insensible a mayúsculas y tildes) con nombres de medio."""
    uniq: dict[str, str] = {}
    for n in list(names) + MEDIA_EXTRA:
        n = _norm_ws(str(n))
        if len(n) >= 4 and n.casefold() not in _GENERIC_NAMES:
            uniq.setdefault(n.casefold(), n)
    alts = sorted(uniq.values(), key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(_accent_insensitive(a) for a in alts) + r")(?!\w)", re.I)


def clean_text(text: str, media_re: "re.Pattern | None" = None, num_token: str = "NUM") -> str:
    """Quita URLs, créditos, firmas, agencias, medios y menciones; unifica cifras en `num_token`.

    En train/dev los números vienen anonimizados como *NUMBER*; en test son cifras reales.
    Unificarlos evita que el formato delate el conjunto de origen.
    """
    for pat in LEAK_PATTERNS.values():
        text = pat.sub(" ", text)
    if media_re is not None:
        text = media_re.sub(" ", text)
    text = NUM_RE.sub(f" {num_token} ", text)
    return _norm_ws(text)


# Tokens + Temas

TOKEN_RE = re.compile(r"[a-záéíóúüñ]+")
COVID_RE = re.compile(
    r"covid|coronavirus|corona virus|pandemi|vacun|cuarentena|confinamiento|sars|cubrebocas|mascarilla|"
    r"contagi|epidemi|\bpcr\b|distanciamiento|inmunidad", re.I)


def tokens(text: str) -> list:
    return TOKEN_RE.findall(text.lower())


# Estilo visual

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
QUOTE_RE = re.compile(r"[\"“”«»]")


def style_features(headline: str, body: str) -> dict:
    """Características de estilo"""
    words = max(len(body.split()), 1)
    per100 = 100.0 / words
    letters_h = [c for c in headline if c.isalpha()]
    letters_b = [c for c in body if c.isalpha()]
    return {
        "titular_palabras": len(headline.split()),
        "titular_mayus_frac": (sum(c.isupper() for c in letters_h) / len(letters_h)) if letters_h else np.nan,
        "titular_todo_mayus": float(bool(letters_h) and headline.upper() == headline),
        "cuerpo_mayus_frac": (sum(c.isupper() for c in letters_b) / len(letters_b)) if letters_b else 0.0,
        "exclamaciones_100p": (body.count("!") + body.count("¡")) * per100,
        "interrogaciones_100p": (body.count("?") + body.count("¿")) * per100,
        "titular_exclama_interroga": float(any(c in headline for c in "!¡?¿")),
        "comillas_100p": len(QUOTE_RE.findall(body)) * per100,
        "cifras_100p": len(NUM_RE.findall(body)) * per100,
        "emojis": float(len(EMOJI_RE.findall(headline + body))),
    }


# Términos discriminativos - ¿Que palabras alejan más fake y true?

def log_odds_z(X_pos, X_neg, alpha: float = 0.5):
    """Log-odds con prior de Dirichlet uniforme y z-score.

    X_pos, X_neg: matrices dispersas de conteos (documentos × términos). Devuelve (log_odds, z).
    
    A mayor z-score, el término analizado es mas discriminativo 
    """
    # Veces que aparece cada término en cada clase (suma por columnas) + suavizado alpha,
    # para evitar divisiones por cero y que los términos raros den valores extremos
    a = np.asarray(X_pos.sum(axis=0)).ravel() + alpha
    b = np.asarray(X_neg.sum(axis=0)).ravel() + alpha
    # Total de palabras de cada clase
    na, nb = a.sum(), b.sum()
    # Log-odds de cada término en una clase menos en la otra: > 0 más propio de pos, < 0 de neg
    delta = np.log(a / (na - a)) - np.log(b / (nb - b))
    # Varianza aproximada de delta: los términos poco frecuentes tienen más incertidumbre
    var = 1.0 / a + 1.0 / b
    # z-score = delta / desviación típica: separa señal real de ruido por baja frecuencia
    return delta, delta / np.sqrt(var)


# Modelos Baseline.

def make_pipe(kind: str = "word", C: float = 1.0):
    """TF-IDF + regresión logística. kind: 'word' (1-2 gramas), 'char' (2-5) o 'domain'."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    if kind == "word":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    elif kind == "char":
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                              max_features=300_000, strip_accents="unicode")
    elif kind == "domain":
        vec = CountVectorizer(token_pattern=r"\S+", lowercase=True, binary=True)
    else:
        raise ValueError(kind)
    return make_pipeline(vec, LogisticRegression(C=C, max_iter=3000, random_state=SEED))
