import pandas as pd

from src.config import FEATURES, TARGET


def validate_data(df: pd.DataFrame) -> None:

    # ==========================================
    # 1. Check required columns
    # ==========================================

    required_columns = set(FEATURES + [TARGET])

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # ==========================================
    # 2. Check empty dataset
    # ==========================================

    if df.empty:
        raise ValueError("Dataset is empty.")

    # ==========================================
    # 3. Check target column
    # ==========================================

    if df[TARGET].isnull().any():
        raise ValueError(
            "Missing values detected in target column."
        )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Target column 'Survived' must contain only 0 or 1."
        )

    # ==========================================
    # 4. Check numerical columns
    # ==========================================

    numerical_columns = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]

    for column in numerical_columns:

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

    # ==========================================
    # 5. Check Pclass
    # ==========================================

    if not df["Pclass"].dropna().between(1, 3).all():
        raise ValueError(
            "Pclass must contain values between 1 and 3."
        )

    # ==========================================
    # 6. Check Age
    # ==========================================

    if (df["Age"].dropna() < 0).any():
        raise ValueError(
            "Age cannot contain negative values."
        )

    # ==========================================
    # 7. Check SibSp
    # ==========================================

    if (df["SibSp"].dropna() < 0).any():
        raise ValueError(
            "SibSp cannot contain negative values."
        )

    # ==========================================
    # 8. Check Parch
    # ==========================================

    if (df["Parch"].dropna() < 0).any():
        raise ValueError(
            "Parch cannot contain negative values."
        )

    # ==========================================
    # 9. Check Fare
    # ==========================================

    if (df["Fare"].dropna() < 0).any():
        raise ValueError(
            "Fare cannot contain negative values."
        )

    # ==========================================
    # 10. Check categorical columns exist
    # ==========================================

    if "Sex" not in df.columns:
        raise ValueError(
            "Column 'Sex' is missing."
        )

    if "Embarked" not in df.columns:
        raise ValueError(
            "Column 'Embarked' is missing."
        )

    print("Data validation checks passed.")