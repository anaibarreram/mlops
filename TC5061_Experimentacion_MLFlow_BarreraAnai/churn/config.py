"""Ubicaciones y parámetros constantes del proyecto."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA = ROOT_DIR / "data" / "processed" / "telco_churn_clean.csv"
EXPERIMENT_NAME = "Telco_Churn_Comparacion"
SEED = 42
TEST_SIZE = 0.20
