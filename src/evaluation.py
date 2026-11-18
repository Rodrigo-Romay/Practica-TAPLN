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


# =====================================================================================
# Ampliación (evaluador común). Lo anterior no se toca: lo usan los notebooks 02 y 03.
# =====================================================================================
import getpass
from datetime import datetime

from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix

SEED = 42
TEST_LOG = Path("results") / "test_evaluations.log"
RANK_FRACS = (0.40, 0.45, 0.50, 0.55, 0.60)

# Dominios satíricos de dev (solo para el desglose descriptivo, nunca como entrada de un modelo).
# Reconstruidos para cuadrar con el EDA: 108 de las 139 Fake de dev (78 %).
SATIRE_DOMAINS = ("elruinaversal.com", "eldizque.com", "censura0.com", "haynoticia.es",
                  "lavozpopular.com", "modonoticia.com", "retroceso.com")


# --- Reglas de decisión ---

def decide_threshold(scores, threshold: float = 0.0) -> np.ndarray:
    """Decisión POR DEFECTO: Fake (1) si score >= umbral del modelo.

    El umbral es el natural del modelo (0 para logit_diff y margen del SVM; 0,5 para
    probabilidades). No depende de las demás noticias ni de la proporción de Fake.
    """
    return predict_threshold(scores, threshold)


def decide_rank(scores, frac_fake: float = 0.5, ids=None) -> np.ndarray:
    """Decisión por RANGO (variante TRANSDUCTIVA): marca Fake la fracción `frac_fake` mejor puntuada.

    Necesita ver todo el conjunto a la vez y supone conocida la proporción de Fake, que en test
    no está documentada: hay que etiquetarla como transductiva y acompañarla de `rank_sensitivity`.
    Empates: se resuelven por `ids` ascendente (determinista y sin depender del orden de las filas);
    sin `ids`, por posición. Siempre se marcan exactamente round(n * frac_fake) noticias.
    """
    scores = np.asarray(scores, dtype=float)
    desempate = np.arange(len(scores)) if ids is None else np.asarray(ids)
    k = int(round(len(scores) * frac_fake))
    orden = np.lexsort((desempate, -scores))   # primero por score descendente, luego por id
    pred = np.zeros(len(scores), dtype=int)
    pred[orden[:k]] = 1
    return pred


# --- Métricas ---

def evaluate_full(y_true, y_pred, scores=None, name: str = "") -> dict:
    """`evaluate` + macro-F1, exactitud y matriz de confusión (tn, fp, fn, tp). Fake = 1 = positiva."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = evaluate(y_true, y_pred, scores, name)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
               accuracy=float((y_true == y_pred).mean()),
               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


def _div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def _f1_parts(y, p):
    """F1 Fake, precisión, recall y macro-F1 vectorizados (filas = réplicas)."""
    tp = (y & p).sum(1).astype(float)
    fp = (~y & p).sum(1).astype(float)
    fn = (y & ~p).sum(1).astype(float)
    tn = (~y & ~p).sum(1).astype(float)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    f1_true = _div(2 * tn, 2 * tn + fp + fn)
    return {"f1_fake": f1, "precision": _div(tp, tp + fp), "recall": _div(tp, tp + fn),
            "macro_f1": (f1 + f1_true) / 2}


def _auc_rows(y, s):
    """AUC por réplica (Mann-Whitney con rangos medios); nan si falta una clase."""
    r = rankdata(s, axis=1)
    n1 = y.sum(1).astype(float)
    n0 = y.shape[1] - n1
    u = (r * y).sum(1) - n1 * (n1 + 1) / 2
    return np.divide(u, n1 * n0, out=np.full_like(u, np.nan), where=(n1 * n0) > 0)


def _boot_metrics(y_true, y_pred, scores, idx):
    y, p = np.asarray(y_true).astype(bool)[idx], np.asarray(y_pred).astype(bool)[idx]
    m = _f1_parts(y, p)
    if scores is not None:
        m["auc"] = _auc_rows(y, np.asarray(scores, dtype=float)[idx])
    return m


def _boot_idx(n, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(y_true, y_pred, scores=None, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = SEED) -> dict:
    """Intervalos de confianza por bootstrap (percentil) remuestreando noticias.

    Devuelve {metrica: (inf, sup)} para f1_fake, precision, recall, macro_f1 y, con `scores`, auc.
    Las predicciones se dejan fijas (no se vuelve a decidir dentro de cada réplica).
    """
    m = _boot_metrics(y_true, y_pred, scores, _boot_idx(len(y_true), n_boot, seed))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {k: tuple(float(x) for x in np.nanpercentile(v, q)) for k, v in m.items()}


def with_ci(row: dict, ci: dict) -> dict:
    """Añade a una fila de métricas las columnas <metrica>_inf y <metrica>_sup."""
    out = dict(row)
    for k, (lo, hi) in ci.items():
        out[f"{k}_inf"], out[f"{k}_sup"] = lo, hi
    return out


def paired_bootstrap(y_true, pred_a, pred_b, scores_a=None, scores_b=None, n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = SEED) -> dict:
    """Comparación pareada A − B con las mismas réplicas para los dos sistemas.

    Devuelve, para F1 Fake y AUC: diferencia observada, IC percentil y p bilateral
    (2 × la menor de las fracciones de réplicas con diferencia <= 0 o >= 0).
    """
    idx = _boot_idx(len(y_true), n_boot, seed)
    ma = _boot_metrics(y_true, pred_a, scores_a, idx)
    mb = _boot_metrics(y_true, pred_b, scores_b, idx)
    ea, eb = evaluate(y_true, pred_a, scores_a), evaluate(y_true, pred_b, scores_b)
    out = {}
    for k in ("f1_fake", "auc"):
        if k not in ma or k not in mb:
            continue
        d = ma[k] - mb[k]
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out.update({f"dif_{k}": float(ea[k] - eb[k]), f"dif_{k}_inf": float(lo),
                    f"dif_{k}_sup": float(hi), f"p_{k}": float(p)})
    return out


# --- Sensibilidad y calibración (solo train/dev) ---

def rank_sensitivity(y_true, scores, fracs=RANK_FRACS, ids=None, name: str = "") -> pd.DataFrame:
    """F1 de la decisión por rango para varias proporciones asumidas de Fake (40–60 %).

    Es informativo: no se elige ninguna proporción con esta tabla.
    """
    rows = []
    for f in fracs:
        r = evaluate_full(y_true, decide_rank(scores, f, ids), scores, name)
        rows.append({**r, "decision": f"rango{int(round(f * 100))} (transductiva)", "frac_asumida": f})
    return pd.DataFrame(rows)


def threshold_leave_group_out(y_true, scores, groups, default: float = 0.0, grid=None) -> pd.DataFrame:
    """Para cada grupo (p. ej. tema): F1 con el umbral por defecto y con el umbral ajustado en el resto.

    Mide si un umbral ajustado se traslada a un tema no visto. SOLO train/dev. `groups` (Topic)
    se usa para partir, nunca como entrada del modelo.
    """
    y, s, g = np.asarray(y_true), np.asarray(scores, dtype=float), np.asarray(groups)
    grid = np.linspace(-10, 10, 81) if grid is None else grid
    rows = []
    for grupo in pd.unique(g):
        m = g == grupo
        t = best_threshold(y[~m], s[~m], grid)
        rows.append({"grupo": grupo, "n": int(m.sum()), "pct_real_fake": float(y[m].mean()),
                     "f1_umbral_defecto": f1_score(y[m], decide_threshold(s[m], default), zero_division=0),
                     "umbral_resto": t,
                     "f1_umbral_resto": f1_score(y[m], decide_threshold(s[m], t), zero_division=0)})
    return pd.DataFrame(rows)


# --- Desgloses descriptivos ---

def breakdown_columns(meta: pd.DataFrame) -> pd.DataFrame:
    """Columnas de desglose a partir de las columnas originales (Headline, Text, Link).

    covid (expresión regular del EDA), longitud del cuerpo, con/sin titular y Fake satírica (dominio).
    Solo para informar: ninguna de estas columnas entra en un modelo.
    """
    import eda_utils as eu
    import preprocessing as pp
    palabras = np.array([len(pp.body_text(h, t).split()) for h, t in zip(meta["Headline"], meta["Text"])])
    return pd.DataFrame({
        "covid": ["COVID" if eu.COVID_RE.search(f"{h} {t}") else "no COVID"
                  for h, t in zip(meta["Headline"], meta["Text"])],
        "longitud": np.select([palabras <= 400, palabras <= 800], ["<=400", "401-800"], ">800"),
        "titular": np.where(meta["Headline"].astype(str).str.strip() == "", "sin titular", "con titular"),
        "satira": meta["Link"].map(eu.extract_domain).isin(SATIRE_DOMAINS).to_numpy(),
    }, index=meta.index)


def evaluate_breakdown(meta: pd.DataFrame, y_true, y_pred, scores=None, name: str = "",
                       satira: bool = False) -> pd.DataFrame:
    """Métricas por subgrupo: COVID / no COVID, longitud y con / sin titular.

    Con satira=True (dev) añade «Fake satíricas» y «Fake no satíricas», cada una frente a TODAS
    las verdaderas (así el AUC mide lo bien que se separa ese tipo de Fake de las noticias reales).
    """
    y, p = np.asarray(y_true), np.asarray(y_pred)
    s = None if scores is None else np.asarray(scores, dtype=float)
    cols = breakdown_columns(meta)
    grupos = [(c, v, (cols[c] == v).to_numpy()) for c in ("covid", "longitud", "titular")
              for v in sorted(cols[c].unique())]
    if satira:
        sat = cols["satira"].to_numpy()
        grupos += [("satira", "Fake satíricas + True", (y == 0) | sat),
                   ("satira", "Fake no satíricas + True", (y == 0) | ~sat)]
    rows = []
    for c, v, m in grupos:
        if m.sum() == 0:
            continue
        r = evaluate_full(y[m], p[m], None if s is None else s[m], name)
        rows.append({"desglose": c, "grupo": v, **r})
    return pd.DataFrame(rows)


# --- Lectura de predicciones y control de uso de test ---

def read_predictions(path, sistema: "str | None" = None, score_col: "str | None" = None,
                     threshold: "float | None" = None) -> pd.DataFrame:
    """Lee un CSV de predicciones en el formato común: Id, pred, score (+ columnas opcionales).

    Si el fichero lleva varios sistemas (columna `sistema`) hay que indicar cuál.
    Para CSV de puntuaciones sin `pred`/`score` (p. ej. los del piloto) se puede dar
    `score_col` (p. ej. 'logit_diff') y `threshold` (0) y se derivan aquí.
    """
    df = pd.read_csv(path)
    if "sistema" in df.columns:
        if sistema is None and df["sistema"].nunique() > 1:
            raise ValueError(f"{path} contiene varios sistemas: indica cuál ({sorted(df['sistema'].unique())})")
        if sistema is not None:
            df = df[df["sistema"] == sistema]
    if score_col:
        df = df.assign(score=df[score_col])
    if "pred" not in df.columns or threshold is not None:
        if "score" not in df.columns or threshold is None:
            raise ValueError("Faltan columnas: se necesita Id, pred y score (o score_col + threshold)")
        df = df.assign(pred=decide_threshold(df["score"], threshold))
    falta = {"Id", "pred", "score"} - set(df.columns)
    if falta:
        raise ValueError(f"Faltan columnas en {path}: {sorted(falta)}")
    if df["Id"].duplicated().any():
        raise ValueError(f"Hay Id repetidos en {path}")
    if not df["pred"].isin([0, 1]).all():
        raise ValueError("`pred` debe ser 0 (True) o 1 (Fake)")
    return df.sort_values("Id").reset_index(drop=True)


def load_labels(root, split: str, final: bool = False) -> pd.DataFrame:
    """Noticias de un conjunto con sus etiquetas. Para test exige final=True (evaluación final)."""
    if split == "test" and not final:
        raise PermissionError("Las etiquetas de test solo se usan en la evaluación final: "
                              "pasa final=True (--final). Cada uso queda registrado.")
    import data_utils as du
    return du.load_clean_splits(Path(root))[split].sort_values("Id").reset_index(drop=True)


def log_test_use(root, sistema: str, quien: "str | None" = None, nota: str = "") -> Path:
    """Añade una línea a results/test_evaluations.log: fecha, sistema, quién y nota."""
    path = Path(root) / TEST_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not path.exists()
    quien = quien or getpass.getuser()
    campos = [datetime.now().isoformat(timespec="seconds"), sistema, quien, nota]
    with path.open("a", encoding="utf-8", newline="\n") as f:
        if nuevo:
            f.write("fecha\tsistema\tquien\tnota\n")
        f.write("\t".join(str(c).replace("\t", " ").replace("\n", " ") for c in campos) + "\n")
    return path


def logged_test_uses(root) -> pd.DataFrame:
    """Evaluaciones sobre test registradas hasta ahora."""
    path = Path(root) / TEST_LOG
    if not path.exists():
        return pd.DataFrame(columns=["fecha", "sistema", "quien", "nota"])
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def _align(labels: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Cruza predicciones y etiquetas por Id; exige que estén todas las noticias del conjunto."""
    faltan = set(labels["Id"]) - set(preds["Id"])
    if faltan:
        raise ValueError(f"Faltan predicciones para {len(faltan)} noticias (p. ej. Id {sorted(faltan)[:5]})")
    return labels.merge(preds[["Id", "pred", "score"]], on="Id", how="left")


def _rank_label(frac_fake: float) -> str:
    return f"rango{int(round(frac_fake * 100))} (transductiva)"


def evaluate_predictions(root, preds: pd.DataFrame, split: str, name: str, final: bool = False,
                         quien: "str | None" = None, nota: str = "", n_boot: int = 2000,
                         frac_fake: float = 0.5) -> dict:
    """Evaluación completa de un sistema en un conjunto.

    Devuelve {'metricas': umbral y rango (transductiva) con IC, 'desglose', 'sensibilidad'}.
    En test exige final=True y registra el uso.
    """
    lab = load_labels(root, split, final)
    d = _align(lab, preds)
    if split == "test":
        log_test_use(root, name, quien, nota)
    y, s, ids = d["label"].to_numpy(), d["score"].to_numpy(), d["Id"].to_numpy()
    decisiones = {"umbral": d["pred"].to_numpy(), _rank_label(frac_fake): decide_rank(s, frac_fake, ids)}
    filas, desg = [], []
    for dec, p in decisiones.items():
        fila = with_ci(evaluate_full(y, p, s, name), bootstrap_ci(y, p, s, n_boot))
        filas.append({**fila, "split": split, "decision": dec})
        desg.append(evaluate_breakdown(d, y, p, s, name, satira=(split == "dev")).assign(split=split, decision=dec))
    return {"metricas": pd.DataFrame(filas), "desglose": pd.concat(desg, ignore_index=True),
            "sensibilidad": rank_sensitivity(y, s, ids=ids, name=name).assign(split=split)}


def compare_predictions(root, sistemas: dict, split: str, final: bool = False, n_boot: int = 2000,
                        frac_fake: float = 0.5) -> pd.DataFrame:
    """Comparación pareada (bootstrap) de todos los pares de `sistemas` = {nombre: predicciones}.

    No registra el uso de test: llamar antes a `evaluate_predictions` para cada sistema.
    """
    lab = load_labels(root, split, final)
    al = {k: _align(lab, v) for k, v in sistemas.items()}
    y, ids = lab["label"].to_numpy(), lab["Id"].to_numpy()
    nombres, rows = list(al), []
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            sa, sb = al[a]["score"].to_numpy(), al[b]["score"].to_numpy()
            for dec, pa, pb in (("umbral", al[a]["pred"].to_numpy(), al[b]["pred"].to_numpy()),
                                (_rank_label(frac_fake), decide_rank(sa, frac_fake, ids),
                                 decide_rank(sb, frac_fake, ids))):
                rows.append({"A": a, "B": b, "split": split, "decision": dec,
                             **paired_bootstrap(y, pa, pb, sa, sb, n_boot)})
    return pd.DataFrame(rows)
