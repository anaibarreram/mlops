"""

IMPORTANTE

Entrenamiento parametrizado con seguimiento completo en MLflow.

Ejecutar sin argumentos:
    python -m churn.modeling.train
Con parámetros:
    python -m churn.modeling.train --c 0.1 --max-iter 1000 --class-weight balanced --seed 42
"""
import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn.config import EXPERIMENT_NAME, RAW_DATA, SEED, TEST_SIZE
from churn.dataset import clean_dataset, get_features_target, read_dataset
from churn.features import make_preprocessor
from churn.plots import save_confusion_matrix


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Debe ser un número mayor que cero")
    return number


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Debe ser un entero mayor que cero")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Telco Churn: entrenar y registrar corridas en MLflow")
    parser.add_argument("--data", type=Path, default=RAW_DATA, help="Ruta al dataset CSV")
    parser.add_argument("--c", type=positive_float, default=1.0, help="Inverso de la regularización (defecto 1.0)")
    parser.add_argument("--max-iter", type=positive_int, default=1000, help="Iteraciones máximas (defecto 1000)")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Pesos de clases")
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla aleatoria (defecto 42)")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Porción de prueba (defecto 0.20)")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="Nombre del experimento")
    parser.add_argument("--run-name", default=None, help="Nombre legible de corrida")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
                        help="Servidor de MLflow (defecto http://127.0.0.1:5000)")
    args = parser.parse_args(argv)
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1")
    return args


def load_data(path):
    return get_features_target(clean_dataset(read_dataset(path)))


def split_data(x, y, test_size, seed):
    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)


def build_pipeline(x_train, c, max_iter, class_weight, seed):
    return Pipeline([
        ("preprocess", make_preprocessor(x_train)),
        ("classifier", LogisticRegression(
            C=c, max_iter=max_iter, class_weight=None if class_weight == "none" else "balanced",
            random_state=seed, solver="liblinear",
        )),
    ])


def evaluate(model, x_test, y_test):
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_churn": float(precision_score(y_test, pred, zero_division=0)),
        "recall_churn": float(recall_score(y_test, pred, zero_division=0)),
        "f1_churn": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    matrix = confusion_matrix(y_test, pred, labels=[0, 1])
    return metrics, matrix


def run_training(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    x, y = load_data(args.data)
    x_train, x_test, y_train, y_test = split_data(x, y, args.test_size, args.seed)
    model = build_pipeline(x_train, args.c, args.max_iter, args.class_weight, args.seed)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        params = {
            "model": "LogisticRegression", "C": args.c, "max_iter": args.max_iter,
            "class_weight": args.class_weight, "seed": args.seed,
            "solver": "liblinear", "test_size": args.test_size,
            "dataset": args.data.name, "train_samples": len(x_train), "test_samples": len(x_test),
        }
        for name, value in params.items():
            mlflow.log_param(name, value)
        model.fit(x_train, y_train)
        metrics, cm = evaluate(model, x_test, y_test)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        with tempfile.TemporaryDirectory() as temp:
            tempdir = Path(temp)
            (tempdir / "metricas.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            save_confusion_matrix(cm, tempdir / "matriz_confusion.png")
            mlflow.log_artifact(str(tempdir / "metricas.json"), artifact_path="evaluacion")
            mlflow.log_artifact(str(tempdir / "matriz_confusion.png"), artifact_path="evaluacion")
        example = x_train.head(3)
        signature = infer_signature(example, model.predict(example))
        mlflow.sklearn.log_model(
            model,
            name="modelo_churn",
            signature=signature,
            input_example=example,
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype"]
        )
        print(f"RUN_ID={run.info.run_id}")
        print(f"EXPERIMENT={args.experiment}")
        print("PARAMETERS=" + json.dumps(params, ensure_ascii=False))
        print("METRICS=" + json.dumps(metrics, sort_keys=True))
        return metrics


def main(argv=None):
    return run_training(parse_args(argv))


if __name__ == "__main__":
    main()
