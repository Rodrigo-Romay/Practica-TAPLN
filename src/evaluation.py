from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def predict_threshold(scores, threshold: float = 0.5) -> np.ndarray:
    " Etiqueta base (Fake) si al puntuación supera el umbral"
    return (np.asarray(scores) >= threshold).astype(int)

def predict_by_rank(scores, frac_fake: float = 0.5) -> np.ndarray:
    """ Marca como Fake el 'frac_fake' de noticias con mayor puntuación.
    A emplear com proporción justificada para balancear el corpus
    
    """""
    scores = np.asarray(scores)
    k = int(round(len(scores) * frac_fake))
    pred = np.zeros(len(scores), dtype=int)
    pred[np.argsort(-scores, kind="stable")[:k]] = 1
    return pred

def best_threshold(y_true, scores, grid=None) -> float:
    """Umbral que maximiza F1 Fake. SOLO para usar con train/dev, nunca con test."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    f1s = [f1_score(y_true, predict_threshold(scores, t), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def evaluate(y_true, y_pred, scores=None, name: str = "") -> dict:
    """Métricas con Fake (1) como clase positiva.

    Devuelve F1, precisión, recall, porcentaje de predicciones Fake y, si hay `scores`, AUC.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "modelo": name,
        "f1_fake": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pct_pred_fake": float(y_pred.mean()),
        "pct_real_fake": float(y_true.mean()),
        "n": len(y_true),
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, scores)
    return out

def results_table(rows: list, round_: int=3) -> pd.DataFrame:
    """List ade dicts de `evaluate` -> tabla ordenada por F1 Fake"""
    df = pd.DataFrame(rows)
    return df.sort_values("f1_fake", ascending=False).round(round_).reset_index(drop=True)


def write_submission(ids, y_pred, path, tag: str = "fakenews") -> None:
    """Escribe el .txt de entrega: una línea por noticia, campos entre comillas y tabuladores.

    Formato: "fakenews"<TAB>"1"<TAB>"True"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for i, p in zip(ids, y_pred):
            f.write(f'"{tag}"\t"{i}"\t"{"Fake" if p == 1 else "True"}"\n')