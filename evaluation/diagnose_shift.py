from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from training.model import ChurnModel


TRAIN_DATA_PATH = "data/processed/features_train"
TEST_DATA_PATH = "data/processed/features_test"

MODEL_PATH = Path("models/churn_model.pt")
REPORT_PATH = Path(
    "models/train_test_distribution_shift.csv"
)

BATCH_SIZE = 8192


def predict_probabilities(
    model: ChurnModel,
    features: np.ndarray,
) -> np.ndarray:
    """Generate churn probabilities in batches."""

    model.eval()

    probability_batches = []

    with torch.no_grad():
        for start_index in range(
            0,
            len(features),
            BATCH_SIZE,
        ):
            end_index = start_index + BATCH_SIZE

            feature_batch = torch.from_numpy(
                features[start_index:end_index]
            )

            logits = model(feature_batch)

            probabilities = torch.sigmoid(logits)

            probability_batches.append(
                probabilities.numpy()
            )

    return np.concatenate(probability_batches)


def main() -> None:
    print("=" * 70)
    print("Train/Test Distribution Shift Diagnosis")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    feature_columns = checkpoint["feature_columns"]
    target_column = "churn"

    required_columns = feature_columns + [target_column]

    print("\nLoading feature datasets...")

    training_dataframe = pd.read_parquet(
        TRAIN_DATA_PATH,
        columns=required_columns,
    )

    testing_dataframe = pd.read_parquet(
        TEST_DATA_PATH,
        columns=required_columns,
    )

    print(
        f"Training rows: "
        f"{len(training_dataframe):,}"
    )

    print(
        f"Testing rows: "
        f"{len(testing_dataframe):,}"
    )

    training_churn_rate = training_dataframe[
        target_column
    ].mean()

    testing_churn_rate = testing_dataframe[
        target_column
    ].mean()

    print("\nTarget distribution:")

    print(
        f"Training churn rate: "
        f"{training_churn_rate:.4f}"
    )

    print(
        f"Testing churn rate:  "
        f"{testing_churn_rate:.4f}"
    )

    # Compare feature statistics.
    training_feature_data = training_dataframe[
        feature_columns
    ]

    testing_feature_data = testing_dataframe[
        feature_columns
    ]

    training_means = training_feature_data.mean()
    testing_means = testing_feature_data.mean()

    training_std = training_feature_data.std().replace(
        0,
        np.nan,
    )

    standardized_mean_shift = (
        testing_means - training_means
    ) / training_std

    # Compare each feature's relationship with churn.
    training_correlations = (
        training_dataframe[
            feature_columns + [target_column]
        ]
        .corr()[target_column]
        .drop(target_column)
    )

    testing_correlations = (
        testing_dataframe[
            feature_columns + [target_column]
        ]
        .corr()[target_column]
        .drop(target_column)
    )

    shift_report = pd.DataFrame(
        {
            "train_mean": training_means,
            "test_mean": testing_means,
            "standardized_mean_shift": (
                standardized_mean_shift
            ),
            "train_correlation_with_churn": (
                training_correlations
            ),
            "test_correlation_with_churn": (
                testing_correlations
            ),
        }
    )

    shift_report["correlation_change"] = (
        shift_report["test_correlation_with_churn"]
        - shift_report[
            "train_correlation_with_churn"
        ]
    )

    shift_report[
        "absolute_correlation_change"
    ] = shift_report["correlation_change"].abs()

    shift_report.index.name = "feature"

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shift_report.to_csv(REPORT_PATH)

    print("\nLargest feature/churn relationship changes:")

    print(
        shift_report.sort_values(
            "absolute_correlation_change",
            ascending=False,
        )[
            [
                "train_correlation_with_churn",
                "test_correlation_with_churn",
                "correlation_change",
            ]
        ]
        .head(10)
        .round(4)
        .to_string()
    )

    # Reconstruct saved model.
    model = ChurnModel(
        input_size=checkpoint["input_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    scaler_mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32,
    )

    scaler_scale = np.asarray(
        checkpoint["scaler_scale"],
        dtype=np.float32,
    )

    scaler_scale = np.where(
        scaler_scale == 0,
        1.0,
        scaler_scale,
    )

    testing_features = testing_feature_data.to_numpy(
        dtype=np.float32
    )

    testing_features = (
        testing_features - scaler_mean
    ) / scaler_scale

    testing_labels = testing_dataframe[
        target_column
    ].to_numpy(dtype=np.int64)

    probabilities = predict_probabilities(
        model=model,
        features=testing_features.astype(
            np.float32
        ),
    )

    threshold = checkpoint.get(
        "prediction_threshold",
        0.5,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        testing_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    print("\nExternal Kaggle test diagnosis:")

    print(
        f"Prediction threshold: {threshold:.2f}"
    )

    print(
        f"Actual churn rate:    "
        f"{testing_labels.mean():.4f}"
    )

    print(
        f"Predicted churn rate: "
        f"{predictions.mean():.4f}"
    )

    print("\nConfusion matrix:")

    print(f"True Negative:  {tn:,}")
    print(f"False Positive: {fp:,}")
    print(f"False Negative: {fn:,}")
    print(f"True Positive:  {tp:,}")

    print("\nThreshold-based metrics:")

    print(
        f"Accuracy:          "
        f"{accuracy_score(testing_labels, predictions):.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy_score(testing_labels, predictions):.4f}"
    )

    print(
        f"Precision:         "
        f"{precision_score(testing_labels, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall:            "
        f"{recall_score(testing_labels, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1:                "
        f"{f1_score(testing_labels, predictions, zero_division=0):.4f}"
    )

    print("\nThreshold-independent metrics:")

    print(
        f"ROC AUC:           "
        f"{roc_auc_score(testing_labels, probabilities):.4f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision_score(testing_labels, probabilities):.4f}"
    )

    print(
        f"\nSaved shift report to: "
        f"{REPORT_PATH}"
    )

    print(
        "\nDistribution diagnosis "
        "completed successfully."
    )


if __name__ == "__main__":
    main()