# Utilidades de análisis del EDA de FakeDeS
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from baselines import make_pipe

# Reexportado: el notebook 01_eda sigue llamando a eu.<nombre> para todo lo movido
from data_utils import SEED, SPLITS, project_root, load_splits 
from preprocessing import (  
    _norm_ws, _accent_insensitive, body_text, model_input, clean_text, build_media_regex,
    LEAK_PATTERNS, URL_RE, NUM_RE, MEDIA_EXTRA,
)


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


