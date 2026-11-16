"""Evaluador en línea de comandos.

Ejemplos (desde la raíz del proyecto):
  python src/evaluate.py --pred results/llm/piloto_dev_gemma4-31b.csv --split dev \
      --nombre gemma4-31b --score-col logit_diff --umbral 0
  python src/evaluate.py --pred results/final/pred_test_gemma4-31b.csv --split test \
      --nombre gemma4-31b --final --quien "Nombre" --entrega results/final/entrega_gemma4-31b.txt

Formato de predicciones: CSV con Id, pred (1 = Fake), score (mayor = más Fake).
Evaluar en test exige --final y deja una línea en results/test_evaluations.log.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du   # noqa: E402
import evaluation as ev   # noqa: E402

COLS = ["modelo", "split", "decision", "f1_fake", "f1_fake_inf", "f1_fake_sup", "precision", "recall",
        "auc", "auc_inf", "auc_sup", "macro_f1", "accuracy", "pct_pred_fake", "tn", "fp", "fn", "tp", "n"]


def _leer(a, path, nombre):
    return ev.read_predictions(path, sistema=a.sistema or (nombre if a.sistema_por_nombre else None),
                               score_col=a.score_col, threshold=a.umbral)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evalúa predicciones de FakeDeS (Fake = clase positiva).")
    ap.add_argument("--pred", required=True, help="CSV de predicciones del sistema A")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--nombre", default=None, help="nombre del sistema (por defecto, el del fichero)")
    ap.add_argument("--comparar", nargs="*", default=[], help="otros CSV para la comparación pareada")
    ap.add_argument("--sistema", default=None, help="valor de la columna `sistema` si el CSV lleva varios")
    ap.add_argument("--sistema-por-nombre", action="store_true", help="usar --nombre como filtro de `sistema`")
    ap.add_argument("--score-col", default=None, help="columna a usar como score (p. ej. logit_diff)")
    ap.add_argument("--umbral", type=float, default=None, help="recalcula pred = score >= umbral")
    ap.add_argument("--frac", type=float, default=0.5, help="proporción de Fake de la decisión por rango")
    ap.add_argument("--boot", type=int, default=2000, help="réplicas de bootstrap")
    ap.add_argument("--final", action="store_true", help="obligatorio para evaluar en test (queda registrado)")
    ap.add_argument("--quien", default=None, help="quién lanza la evaluación final")
    ap.add_argument("--nota", default="", help="nota para el registro de uso de test")
    ap.add_argument("--out", default=None, help="carpeta donde guardar las tablas (opcional)")
    ap.add_argument("--entrega", default=None, help="escribe el .txt de entrega (decisión por umbral)")
    ap.add_argument("--entrega-rango", action="store_true", help="el .txt de entrega usa la decisión por rango")
    a = ap.parse_args(argv)

    root = du.project_root(Path(__file__).resolve().parent)
    nombre = a.nombre or Path(a.pred).stem
    if a.split == "test" and not a.final:
        ap.error("evaluar en test exige --final (se evalúa una sola vez y queda registrado)")

    preds = _leer(a, a.pred, nombre)
    res = ev.evaluate_predictions(root, preds, a.split, nombre, final=a.final, quien=a.quien,
                                  nota=a.nota or f"evaluate.py {a.pred}", n_boot=a.boot, frac_fake=a.frac)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("== Métricas (IC 95 % por bootstrap) ==")
    print(res["metricas"][[c for c in COLS if c in res["metricas"].columns]].round(3).to_string(index=False))
    print("\n== Desglose ==")
    d = res["desglose"]
    print(d[["desglose", "grupo", "decision", "n", "f1_fake", "auc", "pct_pred_fake", "pct_real_fake"]]
          .round(3).to_string(index=False))
    print("\n== Sensibilidad de la decisión por rango (transductiva; informativa) ==")
    print(res["sensibilidad"][["decision", "f1_fake", "precision", "recall", "pct_pred_fake"]]
          .round(3).to_string(index=False))

    comp = None
    if a.comparar:
        sistemas = {nombre: preds}
        for p in a.comparar:
            sistemas[Path(p).stem] = _leer(a, p, Path(p).stem)
        comp = ev.compare_predictions(root, sistemas, a.split, final=a.final, n_boot=a.boot, frac_fake=a.frac)
        print("\n== Comparación pareada (A − B) ==")
        print(comp.round(3).to_string(index=False))

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for k, t in res.items():
            t.to_csv(out / f"{k}_{a.split}_{nombre}.csv", index=False)
        if comp is not None:
            comp.to_csv(out / f"comparacion_{a.split}_{nombre}.csv", index=False)
    if a.entrega:
        pred = ev.decide_rank(preds["score"], a.frac, preds["Id"]) if a.entrega_rango else preds["pred"]
        ev.write_submission(preds["Id"], pred, a.entrega)
        print(f"\nEntrega escrita en {a.entrega}"
              + (" (decisión por rango, transductiva)" if a.entrega_rango else " (decisión por umbral)"))
    return res


if __name__ == "__main__":
    main()
