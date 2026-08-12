import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "ml_dataset.csv"
MODEL_FILE = "ml_model.pkl"


# ============================================================
# FEATURES USED BY THE MODEL
# ============================================================

FEATURE_COLUMNS = [
    "signal",
    "close",
    "SMMA20",
    "SMMA120",
    "smma_difference",
    "smma_difference_percent",
    "price_vs_smma20_percent",
    "price_vs_smma120_percent",
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "candle_range_percent",
    "body_percent",
    "volatility_12",
    "volatility_24",
    "volume",
    "volume_ma_12",
    "volume_ma_24",
    "volume_ratio",
    "smma20_slope",
    "smma120_slope",
    "hour",
    "minute"
]


TARGET_COLUMN = "profitable"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 70)
    print("AI/ML STOCK PREDICTION MODEL")
    print("=" * 70)

    print("\nLoading dataset...")

    try:

        df = pd.read_csv(
            DATASET_FILE
        )

    except FileNotFoundError:

        print(
            f"\nERROR: {DATASET_FILE} not found."
        )

        print(
            "Run feature_engineering.py first."
        )

        raise SystemExit

    print(
        f"Dataset loaded successfully."
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\nPreparing ML data...")

    data = df.copy()

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        print(
            "\nERROR: Missing columns:"
        )

        for column in missing_columns:
            print(
                f"  - {column}"
            )

        raise SystemExit

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    before = len(data)

    data = data.dropna(
        subset=required_columns
    ).copy()

    after = len(data)

    print(
        f"Rows before cleaning: {before}"
    )

    print(
        f"Rows after cleaning: {after}"
    )

    # --------------------------------------------------------
    # Encode signal
    #
    # BUY  = 1
    # SELL = 0
    # --------------------------------------------------------

    data["signal_encoded"] = (
        data["signal"]
        .map({
            "BUY": 1,
            "SELL": 0
        })
    )

    # Check for invalid signals

    invalid_signals = (
        data["signal_encoded"]
        .isna()
        .sum()
    )

    if invalid_signals > 0:

        print(
            f"\nRemoving {invalid_signals} "
            "invalid signal rows."
        )

        data = data[
            data["signal_encoded"]
            .notna()
        ].copy()

    # --------------------------------------------------------
    # Build feature list
    # --------------------------------------------------------

    model_features = [
        "signal_encoded",
        "close",
        "SMMA20",
        "SMMA120",
        "smma_difference",
        "smma_difference_percent",
        "price_vs_smma20_percent",
        "price_vs_smma120_percent",
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "candle_range_percent",
        "body_percent",
        "volatility_12",
        "volatility_24",
        "volume",
        "volume_ma_12",
        "volume_ma_24",
        "volume_ratio",
        "smma20_slope",
        "smma120_slope",
        "hour",
        "minute"
    ]

    X = data[
        model_features
    ].copy()

    y = data[
        TARGET_COLUMN
    ].astype(int)

    return X, y, data, model_features


# ============================================================
# TIME-SERIES TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y, data):

    print("\nCreating chronological train/test split...")

    # Sort according to time before splitting

    if "datetime" in data.columns:

        order = (
            pd.to_datetime(
                data["datetime"]
            )
            .sort_values()
            .index
        )

        X = X.loc[order].reset_index(
            drop=True
        )

        y = y.loc[order].reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # 80% training
    # 20% testing
    # --------------------------------------------------------

    split_index = int(
        len(X) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Random Forest training completed."
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # ROC-AUC requires both classes

    if len(
        np.unique(y_test)
    ) == 2:

        auc = roc_auc_score(
            y_test,
            probabilities
        )

    else:

        auc = None

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    if auc is not None:

        print(
            f"ROC-AUC  : {auc:.4f}"
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Losing Trade",
                "Profitable Trade"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return predictions, probabilities


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model,
    feature_names
):

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importance = pd.DataFrame({

        "feature":
            feature_names,

        "importance":
            model.feature_importances_

    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print(
        importance
        .to_string(index=False)
    )

    return importance


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

def show_sample_predictions(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = model.predict(
        X_test
    )

    sample_count = min(
        10,
        len(X_test)
    )

    results = pd.DataFrame({

        "Actual":
            y_test.iloc[
                :sample_count
            ].values,

        "Prediction":
            predictions[
                :sample_count
            ],

        "Profit_Probability":
            probabilities[
                :sample_count
            ]

    })

    results["Profit_Probability"] = (
        results["Profit_Probability"]
        * 100
    ).round(2)

    print(
        results.to_string(
            index=False
        )
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """
    Save the trained ML model using joblib.
    """

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"Model saved to:\n{MODEL_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Show class distribution
    # --------------------------------------------------------

    print(
        "\nTarget distribution:"
    )

    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Prepare ML data
    # --------------------------------------------------------

    (
        X,
        y,
        data,
        feature_names
    ) = prepare_data(
        df
    )

    print(
        f"\nNumber of ML features:"
        f" {len(feature_names)}"
    )

    print(
        f"Number of samples:"
        f" {len(X)}"
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y,
        data
    )

    # --------------------------------------------------------
    # Show training distribution
    # --------------------------------------------------------

    print(
        "\nTraining target distribution:"
    )

    print(
        y_train.value_counts()
        .sort_index()
    )

    print(
        "\nTesting target distribution:"
    )

    print(
        y_test.value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    show_feature_importance(
        model,
        feature_names
    )

    # --------------------------------------------------------
    # Sample predictions
    # --------------------------------------------------------

    show_sample_predictions(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model
    )

    print("\n" + "=" * 70)
    print("ML MODEL TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()