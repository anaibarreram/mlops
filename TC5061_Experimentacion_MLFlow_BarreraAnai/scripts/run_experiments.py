"""Ejecuta 5 configuraciones únicas con la misma semilla y experimento MLflow.

    python scripts/run_experiments.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ("01_base", 1.0, 1000, "none"),
    ("02_C_0_1", 0.1, 1000, "none"),
    ("03_C_10", 10.0, 1000, "none"),
    ("04_balance", 1.0, 1000, "balanced"),
    ("05_iter_100", 1.0, 100, "none"),
]


def main():
    for name, c, max_iter, class_weight in CONFIGS:
        cmd = [
            sys.executable, "-m", "churn.modeling.train", "--c", str(c),
            "--max-iter", str(max_iter), "--class-weight", class_weight,
            "--seed", "42", "--experiment", "Telco_Churn_Comparacion",
            "--run-name", name,
        ]
        print(f"\n>>> Ejecutando {name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("\nCinco corridas registradas. Abre http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
