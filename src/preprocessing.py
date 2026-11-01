import re
import pandas as pd
import re
import pandas as pd

# --- Texto base (titular + cuerpo) ---

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
    """Regex con nombres de medio."""
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


def media_regex_from_splits(splits: dict) -> re.Pattern:
    """Regex de medios construida con los `Source` de train y dev (nunca de test)."""
    names = pd.concat([splits["train"]["Source"], splits["dev"]["Source"]]).unique()
    return build_media_regex(n for n in names if n)


def prepare(df: pd.DataFrame, media_re: re.Pattern, with_headline: bool = True) -> pd.Series:
    """Texto listo para modelar: sin titular duplicado, limpio y con cifras unificadas.

    with_headline=False usa solo el cuerpo (variante exigida por el 12,6 % de test sin titular).
    """
    build = model_input if with_headline else (lambda h, t: body_text(h, t))
    raw = [build(h, t) for h, t in zip(df["Headline"], df["Text"])]
    return pd.Series([clean_text(x, media_re) for x in raw], index=df.index, name="input")
