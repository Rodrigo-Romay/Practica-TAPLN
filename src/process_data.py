import argparse
from pathlib import Path
import pandas as pd

COLUMNAS = ["Id", "Category", "Topic", "Source", "Headline", "Text", "Link"]


def unificar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Test viene en mayúsculas y con TOPICS en plural -> mismo esquema que train."""
    df = df.rename(columns=lambda c: str(c).strip())
    mapa = {c.upper(): c for c in COLUMNAS}
    mapa["TOPICS"] = "Topic"
    df = df.rename(columns=lambda c: mapa.get(c.upper(), c))
    return df[COLUMNAS]


def unificar_etiqueta(valor) -> str:
    """'Fake'/'True' (strings) o booleanos (False = Fake) -> 'Fake'/'True'."""
    if isinstance(valor, str):
        v = valor.strip().lower()
        if v in ("fake", "false"):
            return "Fake"
        if v in ("true",):
            return "True"
        raise ValueError(f"Etiqueta desconocida: {valor!r}")
    return "True" if bool(valor) else "Fake"


def limpiar(valor) -> str:
    return "" if pd.isna(valor) else " ".join(str(valor).split())


def concatenar(headline: str, text: str) -> str:
    """Titular + texto; si el titular está vacío (caso de test) solo el texto."""
    return f"{headline}\n\n{text}" if headline else text


def cargar(ruta: str) -> pd.DataFrame:
    df = unificar_columnas(pd.read_excel(ruta))
    df["Category"] = df["Category"].map(unificar_etiqueta)
    df["label"] = (df["Category"] == "Fake").astype(int)  # 1 = Fake (clase positiva para F1)
    df["Headline"] = df["Headline"].map(limpiar)
    df["Text"] = df["Text"].map(limpiar)
    df["full_text"] = [concatenar(h, t) for h, t in zip(df["Headline"], df["Text"])]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()

    train, dev, test = cargar(a.train), cargar(a.dev), cargar(a.test)

    # Duplicados entre train y dev: se quitan de dev (train se conserva entero)
    en_train = set(train["Text"])
    dup = dev["Text"].isin(en_train)
    print(f"Duplicados train/dev eliminados de dev: {dup.sum()}")
    dev = dev[~dup].reset_index(drop=True)

    # Duplicados internos (informativo, no se tocan)
    for nombre, df in [("train", train), ("dev", dev), ("test", test)]:
        print(f"{nombre}: {len(df)} filas | Fake={df['label'].sum()} "
              f"True={(df['label'] == 0).sum()} | titulares vacíos="
              f"{(df['Headline'] == '').sum()} | duplicados internos="
              f"{df['Text'].duplicated().sum()}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for nombre, df in [("train", train), ("dev", dev), ("test", test)]:
        df.to_csv(out / f"{nombre}.csv", index=False, encoding="utf-8")
    print(f"Guardado en {out.resolve()}")


if __name__ == "__main__":
    main()