from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "Titanic-Dataset.csv"

MODEL_DIR = ROOT_DIR / "models"

MODEL_PATH = MODEL_DIR / "titanic_model.joblib"

METRICS_PATH = MODEL_DIR / "metrics.json"


FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

TARGET = "Survived"

MIN_ACCURACY = 0.80