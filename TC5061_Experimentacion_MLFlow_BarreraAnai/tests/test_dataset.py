import pandas as pd
import pytest
from churn.dataset import clean_dataset, get_features_target


def test_clean_missing_charges_and_target():
    data = pd.DataFrame({
        "customerID": ["u1", "u2"], "TotalCharges": ["", "98.4"],
        "Churn": ["No", "Yes"], "tenure": [0, 2], "Contract": ["Month-to-month", "One year"],
    })
    cleaned = clean_dataset(data)
    x, y = get_features_target(cleaned)
    assert cleaned["TotalCharges"].tolist() == [0.0, 98.4]
    assert y.tolist() == [0, 1]
    assert "customerID" not in x.columns


def test_duplicate_customer_id_is_rejected():
    data = pd.DataFrame({"customerID": ["u1", "u1"], "TotalCharges": ["1", "2"], "Churn": ["Yes", "No"]})
    with pytest.raises(ValueError, match="duplicados"):
        clean_dataset(data)
