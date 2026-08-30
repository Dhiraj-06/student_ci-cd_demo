import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    DATA_PATH,
    FEATURES,
    METRICS_PATH,
    MIN_ACCURACY,
    MODEL_DIR,
    MODEL_PATH,
    TARGET,
)

from src.data_validation import validate_data


def train_model():

    print("Loading Titanic dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Validating dataset...")

    validate_data(df)

    print("Dataset validation successful.")

    print(f"Total records: {len(df)}")

    X = df[FEATURES]

    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    numerical_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]

    categorical_features = [
        "Sex",
        "Embarked",
    ]

    numerical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("Training Titanic model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print(f"Model Accuracy: {accuracy:.2%}")
    print()

    # ==================================
    # ACCURACY QUALITY GATE
    # ==================================

    if accuracy < MIN_ACCURACY:

        raise RuntimeError(
            f"Quality Gate Failed! "
            f"Accuracy is {accuracy:.2%}, "
            f"but minimum required accuracy "
            f"is {MIN_ACCURACY:.2%}."
        )

    print("Accuracy Quality Gate: PASSED")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"Model saved at: {MODEL_PATH}"
    )

    metrics = {
        "accuracy": round(
            float(accuracy),
            4,
        ),
        "minimum_required_accuracy": MIN_ACCURACY,
        "status": "PASS",
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        f"Metrics saved at: {METRICS_PATH}"
    )

    return accuracy


if __name__ == "__main__":
    train_model()