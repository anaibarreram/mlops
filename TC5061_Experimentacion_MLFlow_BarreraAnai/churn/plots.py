"""Artefacto de evaluación: matriz de confusión exportada como PNG."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(matrix, out_path: Path) -> Path:
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    disp.ax_.set_title("Matriz de confusión | Telco churn")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close("all")
    return out_path
