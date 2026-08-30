import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file does not exist."
        )

    return joblib.load(MODEL_PATH)


def predict_survival(passenger_data):
    model = load_model()

    data = pd.DataFrame(
        [passenger_data],
        columns=FEATURES
    )

    prediction = int(
        model.predict(data)[0]
    )

    probability = float(
        model.predict_proba(data)[0][1]
    )

    return prediction, probability