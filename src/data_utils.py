from pathlib import Path
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

def drop_internal_duplicates(df: pd.DataFrame, col: str = "Text") -> pd.DataFrame:
        return df.drop_duplicates(subset=col, keep="first").reset_index(drop=True)
    

def load_clean_splits(root: Path) -> dict:
    splits = load_splits(root)
    splits["dev"] = drop_internal_duplicates(splits["dev"])
    return splits

def get_xy(df: pd.DataFrame, text_col: str = "input" ) -> tuple:
    return df[text_col].to_list(), df["label"].to_numpy()

def group_ids(root: Path, df: pd.DataFrame):
    """Grupo de cada fila de train (para GroupKFold), alineado por Id."""
    g = pd.read_csv(root / "results" / "eda" / "10_train_grupos.csv")
    return df[["Id"]].merge(g, on="Id", how="left")["grupo"].to_numpy()
