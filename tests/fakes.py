"""Dobles de prueba: tokenizador y modelo diminutos, y un proyecto sintético en una carpeta temporal.

El «modelo» puntúa Fake si en la noticia (último mensaje de usuario) hay más «bulo» que «oficial».
No sustituye a ninguna prueba con modelos reales: solo comprueba que las piezas encajan.
"""
import numpy as np
import pandas as pd
import torch


class _Enc(dict):
    def to(self, device):
        return self


class FakeTok:
    def __init__(self, size: int = 20000):
        self.vocab, self.size = {}, size

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 1)

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.split()]

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(int(i), "?") for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return " ".join(f"<{m['role']}> {m['content']}" for m in messages) + " <assistant>"

    def __call__(self, prompt, return_tensors=None, add_special_tokens=False):
        ids = self.encode(prompt)
        return _Enc(input_ids=torch.tensor([ids]) if return_tensors == "pt" else ids)


class _Out:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    device = "cpu"

    def __init__(self, tok):
        self.tok, self.llamadas = tok, 0

    def __call__(self, input_ids, logits_to_keep=None, **kw):
        self.llamadas += 1
        ids = input_ids[0].tolist()
        ini = len(ids) - 1 - ids[::-1].index(self.tok._id("<user>"))   # solo la noticia, no los ejemplos
        trozo = ids[ini:]
        s = 2.0 * (trozo.count(self.tok._id("bulo")) - trozo.count(self.tok._id("oficial")))
        logits = torch.zeros(1, 1, self.tok.size)
        logits[0, 0, self.tok._id("Falsa")] = 5 + s
        logits[0, 0, self.tok._id("Verdadera")] = 5 - s
        return _Out(logits)


def fake_loader(name, quant=None):
    tok = FakeTok()
    return tok, FakeModel(tok)


RELLENO = ("gobierno ciudad personas informe semana datos medida plan anuncio reunión cambio grupo "
           "estudio caso país mercado equipo proyecto servicio centro historia lugar momento").split()


def make_texts(n: int, seed: int, ruido: float = 0.15) -> pd.DataFrame:
    """n noticias sintéticas, mitad Fake. Con probabilidad `ruido` la palabra clave va cambiada."""
    rng = np.random.default_rng(seed)
    filas = []
    for i in range(1, n + 1):
        y = i % 2
        clave = "bulo" if (y == 1) != (rng.random() < ruido) else "oficial"
        frases = [" ".join(rng.choice(RELLENO, 8)).capitalize() + "." for _ in range(4)]
        frases.insert(int(rng.integers(0, 5)), f"Según el comunicado {clave} {clave} del martes {i * 7 + seed}.")
        filas.append({"Id": i, "Category": "Fake" if y else "True", "Topic": ["A", "B", "C"][i % 3],
                      "Source": f"medio{i % 4}", "Headline": "" if i % 5 == 0 else f"Titular {seed} {i}",
                      "Text": " ".join(frases), "Link": f"https://medio{i % 4}.com/n{i}", "label": y})
    df = pd.DataFrame(filas)
    df["full_text"] = df["Headline"] + "\n\n" + df["Text"]
    return df


def make_project(root, n_train: int = 40, n_dev: int = 24, n_test: int = 24):
    """Proyecto mínimo: data/processed/*.csv y src/ (para que `project_root` lo reconozca)."""
    (root / "data" / "processed").mkdir(parents=True)
    (root / "src").mkdir()
    for split, n, seed in (("train", n_train, 1), ("dev", n_dev, 2), ("test", n_test, 3)):
        make_texts(n, seed).to_csv(root / "data" / "processed" / f"{split}.csv", index=False)
    return root
