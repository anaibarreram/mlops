"""Predicción de churn desde el modelo registrado; útil para comprobar la carga."""
import argparse
import mlflow.sklearn
from churn.dataset import clean_dataset, get_features_target, read_dataset


def main():
    parser = argparse.ArgumentParser(description="Predice churn desde un modelo registrado en MLflow")
    parser.add_argument("--model-uri", required=True, help="URI en MLflow: models:/<model_id> o runs:/...")
    parser.add_argument("--data", required=True, help="CSV de clientes")
    args = parser.parse_args()
    model = mlflow.sklearn.load_model(args.model_uri)
    x, _ = get_features_target(clean_dataset(read_dataset(args.data)))
    predictions = model.predict(x)
    print(f"Predicciones generadas: {len(predictions)}; churn estimado: {int(predictions.sum())}")


if __name__ == "__main__":
    main()
