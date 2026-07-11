import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.model import ChurnModel


TRAIN_DATA_PATH = "data/processed/features_train"
EXTERNAL_TEST_DATA_PATH = "data/processed/features_test"

MODEL_OUTPUT_PATH = Path("models/churn_model.pt")
METRICS_OUTPUT_PATH = Path("models/metrics.json")

RANDOM_SEED = 42

BATCH_SIZE = 2048
EPOCHS = 8
LEARNING_RATE = 0.002
PREDICTION_THRESHOLD = 0.5

# Kaggle training-master split:
# 70% training
# 15% validation
# 15% internal test
HOLDOUT_SIZE = 0.30
INTERNAL_TEST_RATIO_WITHIN_HOLDOUT = 0.50


FEATURE_COLUMNS = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "total_spend",
    "last_interaction",
    "gender_male",
    "gender_female",
    "subscription_basic",
    "subscription_standard",
    "subscription_premium",
    "contract_monthly",
    "contract_quarterly",
    "contract_annual",
]

TARGET_COLUMN = "churn"


def set_random_seeds() -> None:
    """Make model training more reproducible."""

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def get_device() -> torch.device:
    """Select CUDA, Apple Silicon MPS, or CPU."""

    requested_device = os.getenv("DEVICE")

    if requested_device:
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def load_dataset(path: str) -> pd.DataFrame:
    """Load feature Parquet data into a Pandas DataFrame."""

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    dataframe = pd.read_parquet(
        path,
        columns=required_columns,
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    if dataframe.isnull().any().any():
        raise ValueError(
            f"Dataset contains null values: {path}"
        )

    return dataframe


def create_data_loader(
    features: np.ndarray,
    labels: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    """Convert NumPy arrays into a PyTorch DataLoader."""

    feature_tensor = torch.from_numpy(
        features.astype(np.float32)
    )

    label_tensor = torch.from_numpy(
        labels.astype(np.float32)
    )

    dataset = TensorDataset(
        feature_tensor,
        label_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def train_one_epoch(
    model: ChurnModel,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    total_examples = 0

    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        current_batch_size = labels.size(0)

        total_loss += (
            loss.item() * current_batch_size
        )

        total_examples += current_batch_size

    return total_loss / total_examples


def evaluate_model(
    model: ChurnModel,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model loss and classification metrics."""

    model.eval()

    total_loss = 0.0
    total_examples = 0

    probability_batches = []
    label_batches = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            probabilities = torch.sigmoid(logits)

            current_batch_size = labels.size(0)

            total_loss += (
                loss.item() * current_batch_size
            )

            total_examples += current_batch_size

            probability_batches.append(
                probabilities.cpu().numpy()
            )

            label_batches.append(
                labels.cpu().numpy()
            )

    probabilities = np.concatenate(
        probability_batches
    )

    labels = np.concatenate(
        label_batches
    )

    predictions = (
        probabilities >= PREDICTION_THRESHOLD
    ).astype(np.int64)

    return {
        "loss": float(
            total_loss / total_examples
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
    }


def print_metrics(
    title: str,
    metrics: dict[str, float],
) -> None:
    """Print formatted evaluation metrics."""

    print(title)

    print(
        f"Loss: {metrics['loss']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"ROC AUC: {metrics['roc_auc']:.4f} | "
        f"Average Precision: "
        f"{metrics['average_precision']:.4f}"
    )


def main() -> None:
    set_random_seeds()

    device = get_device()

    print("=" * 70)
    print("PyTorch Customer Churn Training")
    print("=" * 70)
    print(f"Using device: {device}")

    print("\nLoading Parquet feature datasets...")

    training_dataframe = load_dataset(
        TRAIN_DATA_PATH
    )

    external_test_dataframe = load_dataset(
        EXTERNAL_TEST_DATA_PATH
    )

    print(
        f"Kaggle training-master rows: "
        f"{len(training_dataframe):,}"
    )

    print(
        f"Kaggle external test rows: "
        f"{len(external_test_dataframe):,}"
    )

    full_training_features = training_dataframe[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    full_training_labels = training_dataframe[
        TARGET_COLUMN
    ].to_numpy(dtype=np.float32)

    external_test_features = (
        external_test_dataframe[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32)
    )

    external_test_labels = (
        external_test_dataframe[
            TARGET_COLUMN
        ].to_numpy(dtype=np.float32)
    )

    # First split:
    # 70% model training
    # 30% temporary holdout
    (
        training_features,
        holdout_features,
        training_labels,
        holdout_labels,
    ) = train_test_split(
        full_training_features,
        full_training_labels,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_SEED,
        stratify=full_training_labels,
    )

    # Second split:
    # 15% validation
    # 15% internal test
    (
        validation_features,
        internal_test_features,
        validation_labels,
        internal_test_labels,
    ) = train_test_split(
        holdout_features,
        holdout_labels,
        test_size=INTERNAL_TEST_RATIO_WITHIN_HOLDOUT,
        random_state=RANDOM_SEED,
        stratify=holdout_labels,
    )

    print("\nDataset split:")

    print(
        f"Model training rows: "
        f"{len(training_features):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_features):,}"
    )

    print(
        f"Internal test rows: "
        f"{len(internal_test_features):,}"
    )

    print(
        f"External test rows: "
        f"{len(external_test_features):,}"
    )

    # Fit the scaler only on model training data.
    # This prevents validation and test leakage.
    scaler = StandardScaler()

    training_features = scaler.fit_transform(
        training_features
    ).astype(np.float32)

    validation_features = scaler.transform(
        validation_features
    ).astype(np.float32)

    internal_test_features = scaler.transform(
        internal_test_features
    ).astype(np.float32)

    external_test_features = scaler.transform(
        external_test_features
    ).astype(np.float32)

    training_loader = create_data_loader(
        training_features,
        training_labels,
        shuffle=True,
    )

    validation_loader = create_data_loader(
        validation_features,
        validation_labels,
        shuffle=False,
    )

    internal_test_loader = create_data_loader(
        internal_test_features,
        internal_test_labels,
        shuffle=False,
    )

    external_test_loader = create_data_loader(
        external_test_features,
        external_test_labels,
        shuffle=False,
    )

    model = ChurnModel(
        input_size=len(FEATURE_COLUMNS)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_f1 = -1.0
    best_epoch = 0
    best_model_state = None
    best_validation_metrics = None

    print("\nStarting model training...\n")

    for epoch in range(1, EPOCHS + 1):
        training_loss = train_one_epoch(
            model=model,
            data_loader=training_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_metrics = evaluate_model(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {training_loss:.4f} | "
            f"Val Loss: "
            f"{validation_metrics['loss']:.4f} | "
            f"Val Accuracy: "
            f"{validation_metrics['accuracy']:.4f} | "
            f"Val F1: "
            f"{validation_metrics['f1']:.4f} | "
            f"Val ROC AUC: "
            f"{validation_metrics['roc_auc']:.4f}"
        )

        if (
            validation_metrics["f1"]
            > best_validation_f1
        ):
            best_validation_f1 = (
                validation_metrics["f1"]
            )

            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            best_validation_metrics = (
                validation_metrics.copy()
            )

    if best_model_state is None:
        raise RuntimeError(
            "Training did not produce a model."
        )

    model.load_state_dict(
        best_model_state
    )

    print(
        f"\nBest model came from epoch {best_epoch}."
    )

    internal_test_metrics = evaluate_model(
        model=model,
        data_loader=internal_test_loader,
        criterion=criterion,
        device=device,
    )

    external_test_metrics = evaluate_model(
        model=model,
        data_loader=external_test_loader,
        criterion=criterion,
        device=device,
    )

    print()

    print_metrics(
        "Internal test metrics:",
        internal_test_metrics,
    )

    print()

    print_metrics(
        "External Kaggle test metrics:",
        external_test_metrics,
    )

    MODEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_state_cpu = {
        name: tensor.detach().cpu()
        for name, tensor
        in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": model_state_cpu,
        "input_size": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "prediction_threshold": (
            PREDICTION_THRESHOLD
        ),
        "best_epoch": best_epoch,
        "validation_metrics": (
            best_validation_metrics
        ),
        "internal_test_metrics": (
            internal_test_metrics
        ),
        "external_test_metrics": (
            external_test_metrics
        ),
    }

    torch.save(
        checkpoint,
        MODEL_OUTPUT_PATH,
    )

    metrics_output = {
        "best_epoch": best_epoch,
        "validation": best_validation_metrics,
        "internal_test": internal_test_metrics,
        "external_test": external_test_metrics,
        "parameters": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "prediction_threshold": (
                PREDICTION_THRESHOLD
            ),
            "input_features": len(
                FEATURE_COLUMNS
            ),
            "random_seed": RANDOM_SEED,
        },
        "dataset_sizes": {
            "training": len(training_features),
            "validation": len(
                validation_features
            ),
            "internal_test": len(
                internal_test_features
            ),
            "external_test": len(
                external_test_features
            ),
        },
    }

    with METRICS_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics_output,
            metrics_file,
            indent=2,
        )

    print(
        f"\nSaved model to: "
        f"{MODEL_OUTPUT_PATH}"
    )

    print(
        f"Saved metrics to: "
        f"{METRICS_OUTPUT_PATH}"
    )

    print(
        "\nPyTorch training "
        "completed successfully."
    )


if __name__ == "__main__":
    main()