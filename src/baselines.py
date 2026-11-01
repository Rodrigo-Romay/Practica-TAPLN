import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from data_utils import SEED


def make_vectorizer(kind: str = "word"):
    """'word' (1-2 gramas), 'char' (2-5) o 'domain'."""
    if kind == "word":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    if kind == "char":
        return TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                               max_features=300_000, strip_accents="unicode")
    if kind == "domain":
        return CountVectorizer(token_pattern=r"\S+", lowercase=True, binary=True)
    raise ValueError(kind)


def make_classifier(clf: str = "lr", C: float = 1.0):
    """'lr' regresión logística, 'svm' LinearSVC, 'cnb' ComplementNB, 'dummy_fake' / 'dummy_true'."""
    if clf == "lr":
        return LogisticRegression(C=C, max_iter=3000, random_state=SEED)
    if clf == "svm":
        return LinearSVC(C=C, random_state=SEED)
    if clf == "cnb":
        return ComplementNB(alpha=C)  # aquí C hace de alpha (suavizado)
    if clf == "dummy_fake":
        return DummyClassifier(strategy="constant", constant=1)
    if clf == "dummy_true":
        return DummyClassifier(strategy="constant", constant=0)
    raise ValueError(clf)


def make_pipe(kind: str = "word", C: float = 1.0, clf: str = "lr"):
    """Vectorizador TF-IDF + clasificador. Con clf='lr' equivale a la versión del análisis exploratorio."""
    return make_pipeline(make_vectorizer(kind), make_classifier(clf, C))


def get_scores(model, X) -> np.ndarray:
    """Puntuación de Fake de cualquier modelo: probabilidad si existe, margen si no."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def cv_scores(pipe, X, y, groups, n_splits: int = 5) -> np.ndarray:
    """Puntuaciones fuera de pliegue con GroupKFold (grupos de results/eda/10_train_grupos.csv)."""
    method = "predict_proba" if hasattr(pipe[-1], "predict_proba") else "decision_function"
    out = cross_val_predict(pipe, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits), method=method)
    return out[:, 1] if out.ndim == 2 else out
