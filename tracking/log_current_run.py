import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature

from training.model import ChurnModel


MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
EXPERIMENT_NAME = "Customer Churn Prediction"
RUN_NAME = "mlp-lr-002-v2"

MODEL_PATH = Path("models/churn_model.pt")
METRICS_PATH = Path("models/metrics.json")
SHIFT_REPORT_PATH = Path(
    "models/train_test_distribution_shift.csv"
)

REGISTERED_MODEL_NAME = "CustomerChurnModel"


def validate_files() -> None:
    """Ensure required model output files exist."""

    required_files = [
        MODEL_PATH,
        METRICS_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files: {missing_files}"
        )


def load_metrics() -> dict:
    """Load previously generated evaluation metrics."""

    with METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as metrics_file:
        return json.load(metrics_file)


def load_model() -> tuple[
    ChurnModel,
    dict,
]:
    """Reconstruct the trained PyTorch model."""

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = ChurnModel(
        input_size=checkpoint["input_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def flatten_metrics(
    split_name: str,
    split_metrics: dict,
) -> dict[str, float]:
    """Add the dataset split name to each metric."""

    return {
        f"{split_name}_{metric_name}": float(value)
        for metric_name, value
        in split_metrics.items()
    }


def main() -> None:
    validate_files()

    metrics_data = load_metrics()
    model, checkpoint = load_model()

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    input_size = checkpoint["input_size"]

    # Example input used to define the model input format.
    input_example = np.zeros(
        shape=(2, input_size),
        dtype=np.float32,
    )

    with torch.no_grad():
        output_example = model(
            torch.from_numpy(input_example)
        ).numpy()

    model_signature = infer_signature(
        input_example,
        output_example,
    )

    with mlflow.start_run(
        run_name=RUN_NAME
    ) as run:
        parameters = {
            **metrics_data["parameters"],
            "best_epoch": metrics_data[
                "best_epoch"
            ],
            "model_architecture": "15-32-16-1",
            "optimizer": "Adam",
            "loss_function": (
                "BCEWithLogitsLoss"
            ),
            "dataset_source": "Kaggle",
            "feature_pipeline": "PySpark",
        }

        mlflow.log_params(parameters)

        dataset_size_parameters = {
            f"{split_name}_rows": row_count
            for split_name, row_count
            in metrics_data[
                "dataset_sizes"
            ].items()
        }

        mlflow.log_params(
            dataset_size_parameters
        )

        validation_metrics = flatten_metrics(
            "validation",
            metrics_data["validation"],
        )

        internal_test_metrics = flatten_metrics(
            "internal_test",
            metrics_data["internal_test"],
        )

        external_test_metrics = flatten_metrics(
            "external_test",
            metrics_data["external_test"],
        )

        mlflow.log_metrics(
            validation_metrics
        )

        mlflow.log_metrics(
            internal_test_metrics
        )

        mlflow.log_metrics(
            external_test_metrics
        )

        mlflow.set_tags(
            {
                "framework": "PyTorch",
                "data_processing": "PySpark",
                "task": "binary_classification",
                "target": "customer_churn",
                "evaluation_design": (
                    "validation_internal_external"
                ),
                "distribution_shift_detected": (
                    "true"
                ),
            }
        )

        # Log saved reports and checkpoint.
        mlflow.log_artifact(
            str(METRICS_PATH),
            artifact_path="reports",
        )

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="checkpoints",
        )

        if SHIFT_REPORT_PATH.exists():
            mlflow.log_artifact(
                str(SHIFT_REPORT_PATH),
                artifact_path="reports",
            )

        source_files = [
            Path("training/model.py"),
            Path("training/train.py"),
            Path("spark/preprocess.py"),
            Path("spark/build_features.py"),
        ]

        for source_file in source_files:
            if source_file.exists():
                mlflow.log_artifact(
                    str(source_file),
                    artifact_path="source_code",
                )

        # Log a complete MLflow PyTorch model and
        # create a registered model version.
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="churn_model",
            input_example=input_example,
            signature=model_signature,
            code_paths=["training"],
            registered_model_name=(
                REGISTERED_MODEL_NAME
            ),
            serialization_format="pickle",
        )

        print("=" * 70)
        print("MLflow logging completed successfully.")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Run name: {RUN_NAME}")
        print(f"Run ID: {run.info.run_id}")
        print(
            f"Registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )
        print("=" * 70)


if __name__ == "__main__":
    main()