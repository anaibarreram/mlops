import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from churn.features import make_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def test_same_seed_returns_same_f1_on_fixed_data():
    rng = np.random.default_rng(123)
    n = 300
    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, n),
        "MonthlyCharges": rng.normal(60, 20, n),
        "Contract": rng.choice(["One year", "Month-to-month"], n),
    })
    y = ((df["tenure"] < 30) & (df["Contract"] == "Month-to-month")).astype(int)
    scores = []
    for _ in range(2):
        xtr, xt, ytr, yt = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
        model = Pipeline([
            ("preprocess", make_preprocessor(xtr)),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="liblinear")),
        ])
        model.fit(xtr, ytr)
        scores.append(f1_score(yt, model.predict(xt)))
    assert scores[0] == scores[1]
