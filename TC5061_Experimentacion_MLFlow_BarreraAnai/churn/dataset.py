"""Lectura y depuración de datos; no ajusta transformadores estadísticos."""
from pathlib import Path
import pandas as pd

TARGET = "Churn"
IDENTIFIER = "customerID"


def read_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró {path}. Copia el CSV Telco en data/raw/ "
            "(consulta las instrucciones del README)."
        )
    df = pd.read_csv(path)
    required = {TARGET, IDENTIFIER, "TotalCharges"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas requeridas ausentes: {sorted(missing)}")
    if df.empty:
        raise ValueError("El archivo CSV no tiene registros")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Replica la decisión de la práctica base: cargos vacíos se sustituyen por cero."""
    data = df.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0.0)
    labels = data[TARGET].astype(str).str.strip().str.lower()
    if not labels.isin(["yes", "no"]).all():
        raise ValueError("La columna Churn debe contener exclusivamente Yes y No")
    data[TARGET] = labels.map({"no": 0, "yes": 1}).astype(int)
    if data[IDENTIFIER].duplicated().any():
        raise ValueError("Se detectaron customerID duplicados; revisar el dataset")
    return data


def get_features_target(data: pd.DataFrame):
    return data.drop(columns=[TARGET, IDENTIFIER]), data[TARGET].astype(int)
