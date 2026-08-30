import json

from src.config import (
    FEATURES,
    METRICS_PATH,
    MIN_ACCURACY,
    MODEL_PATH,
)

from src.predict import (
    load_model,
    predict_survival,
)


# ==========================================
# TEST 1: Model file exists
# ==========================================

def test_model_file_exists():

    assert MODEL_PATH.exists(), (
        "Titanic model file is missing."
    )


# ==========================================
# TEST 2: Metrics file exists
# ==========================================

def test_metrics_file_exists():

    assert METRICS_PATH.exists(), (
        "Metrics file is missing."
    )


# ==========================================
# TEST 3: Accuracy quality gate
# ==========================================

def test_accuracy_quality_gate():

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metrics = json.load(file)

    accuracy = metrics["accuracy"]

    assert accuracy >= MIN_ACCURACY, (
        f"Accuracy {accuracy:.2%} is below "
        f"required accuracy {MIN_ACCURACY:.2%}."
    )


# ==========================================
# TEST 4: Model can be loaded
# ==========================================

def test_model_can_be_loaded():

    model = load_model()

    assert model is not None


# ==========================================
# TEST 5: Custom passenger prediction
# ==========================================

def test_custom_passenger_prediction():

    passenger = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80,
        "Embarked": "C",
    }

    prediction, probability = (
        predict_survival(passenger)
    )

    assert prediction in [0, 1]

    assert 0 <= probability <= 1


# ==========================================
# TEST 6: Second custom passenger
# ==========================================

def test_second_custom_passenger():

    passenger = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 40,
        "SibSp": 1,
        "Parch": 2,
        "Fare": 10,
        "Embarked": "S",
    }

    prediction, probability = (
        predict_survival(passenger)
    )

    assert prediction in [0, 1]

    assert 0 <= probability <= 1


# ==========================================
# TEST 7: Required features
# ==========================================

def test_required_features():

    expected_features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]

    assert FEATURES == expected_features