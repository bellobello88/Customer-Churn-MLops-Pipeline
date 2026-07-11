# Customer Churn Prediction MLOps Pipeline

An end-to-end machine learning pipeline for predicting customer churn using **PySpark**, **PyTorch**, and **MLflow**.

The project demonstrates how customer data can move from raw CSV files through scalable preprocessing, feature engineering, neural-network training, model evaluation, experiment tracking, and model registration.

## Project Highlights

- Processed more than **500,000 customer records** with PySpark
- Converted cleaned datasets from CSV to partitioned Parquet
- Built reusable feature-engineering pipelines for numerical and categorical data
- Trained a binary classification neural network with PyTorch
- Evaluated the model on validation, internal test, and external test datasets
- Diagnosed train/test distribution shift
- Logged parameters, metrics, artifacts, source code, and model versions with MLflow
- Registered the trained model in the MLflow Model Registry

## Architecture

```text
Kaggle Customer Churn CSV
          |
          v
      PySpark ETL
          |
          v
 Cleaned Parquet Datasets
          |
          v
PySpark Feature Engineering
          |
          v
 Numeric Feature Parquet
          |
          v
   PyTorch MLP Training
          |
          +-------------------+
          |                   |
          v                   v
 Internal Evaluation    External Evaluation
          |                   |
          +---------+---------+
                    |
                    v
          MLflow Experiment Tracking
                    |
                    v
          MLflow Model Registry
```

## Technology Stack

| Technology | Purpose |
|---|---|
| PySpark | Data cleaning, schema enforcement, validation, feature engineering, and Parquet output |
| PyTorch | Neural-network training and binary churn classification |
| MLflow | Experiment tracking, metric logging, artifact storage, and model registration |
| Pandas / PyArrow | Loading processed Parquet data for model training |
| scikit-learn | Dataset splitting, feature scaling, and evaluation metrics |
| Python | Pipeline implementation |
| Java 17 | Local Spark runtime |

## Dataset

This project uses the Kaggle **Customer Churn Dataset**, which provides separate training and testing CSV files.
https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?resource=download

### Raw Features

- Customer ID
- Age
- Gender
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Subscription Type
- Contract Length
- Total Spend
- Last Interaction
- Churn

### Target

```text
Churn = 0  -> customer retained
Churn = 1  -> customer churned
```

The raw dataset files are not committed to this repository. Download them from Kaggle and place them inside:

```text
data/raw/
├── customer_churn_dataset-training-master.csv
└── customer_churn_dataset-testing-master.csv
```

## Data Pipeline

### 1. PySpark ETL

The ETL pipeline:

- Applies an explicit Spark schema
- Renames columns to snake_case
- Handles missing feature values
- Removes invalid rows
- Removes duplicate customer IDs
- Validates binary churn labels
- Writes cleaned data to Parquet

Run:

```bash
python spark/preprocess.py
```

Output:

```text
data/processed/churn_train/
data/processed/churn_test/
```

### 2. Feature Engineering

The feature pipeline converts categorical columns into numeric one-hot features.

Examples:

```text
gender
-> gender_male
-> gender_female

subscription_type
-> subscription_basic
-> subscription_standard
-> subscription_premium

contract_length
-> contract_monthly
-> contract_quarterly
-> contract_annual
```

The final model input contains **15 numeric features**.

Run:

```bash
python spark/build_features.py
```

Output:

```text
data/processed/features_train/
data/processed/features_test/
```

## Model

The churn classifier is a multilayer perceptron:

```text
15 input features
        |
        v
Linear(15, 32)
        |
       ReLU
        |
   Dropout(0.2)
        |
Linear(32, 16)
        |
       ReLU
        |
Linear(16, 1)
```

Training configuration:

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss |
| Batch size | 2048 |
| Epochs | 8 |
| Baseline learning rate | 0.001 |
| Prediction threshold | 0.5 |
| Random seed | 42 |

Run:

```bash
python -m training.train
```

The best model is selected using validation F1 score.

## Evaluation Design

The Kaggle training dataset is split into:

```text
70% training
15% validation
15% internal test
```

The separate Kaggle testing dataset is treated as an **external test set**.

| Dataset split | Rows |
|---|---:|
| Training | 308,582 |
| Validation | 66,125 |
| Internal test | 66,125 |
| External test | 64,374 |

This design separates normal in-distribution performance from performance under dataset shift.

## Baseline Results

### Validation

| Metric | Score |
|---|---:|
| Accuracy | 0.9892 |
| Precision | 0.9990 |
| Recall | 0.9819 |
| F1 | 0.9904 |
| ROC AUC | 0.9984 |
| Average Precision | 0.9991 |

### Internal Test

| Metric | Score |
|---|---:|
| Accuracy | 0.9897 |
| Precision | 0.9993 |
| Recall | 0.9825 |
| F1 | 0.9908 |
| ROC AUC | 0.9983 |
| Average Precision | 0.9990 |

### External Test

| Metric | Score |
|---|---:|
| Accuracy | 0.5134 |
| Precision | 0.4932 |
| Recall | 0.9974 |
| F1 | 0.6601 |
| ROC AUC | 0.6442 |
| Average Precision | 0.5652 |

## Distribution Shift Analysis

The model performs strongly on the internal test set but degrades substantially on the external Kaggle test set.

The external model predictions show:

```text
Actual churn rate:    47.37%
Predicted churn rate: 95.82%
```

Important feature relationships also changed between the training and external test datasets.

Examples:

| Feature | Train correlation with churn | External correlation with churn |
|---|---:|---:|
| Contract Monthly | 0.4336 | 0.0615 |
| Total Spend | -0.4294 | -0.0789 |
| Support Calls | 0.5743 | 0.3046 |
| Tenure | -0.0519 | 0.1953 |
| Payment Delay | 0.3121 | 0.5574 |

This indicates that the external performance issue is not only caused by the classification threshold. The relationship between customer features and churn changes across datasets.

Run the diagnosis:

```bash
python -m evaluation.diagnose_shift
```

Output:

```text
models/train_test_distribution_shift.csv
```

## MLflow Experiment Tracking

MLflow records:

- Learning rate
- Epoch count
- Batch size
- Model architecture
- Validation metrics
- Internal test metrics
- External test metrics
- PyTorch checkpoint
- Evaluation reports
- Pipeline source code
- Registered model versions

Start the local MLflow server:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

In a second terminal, log the current run:

```bash
python -m tracking.log_current_run
```

The experiment is stored under:

```text
Customer Churn Prediction
```

The registered model is:

```text
CustomerChurnModel
```

## Repository Structure

```text
customer-churn-mlops-pipeline/
├── data/
│   ├── raw/                         # Kaggle CSV files, not committed
│   └── processed/                   # Generated Parquet files, not committed
├── spark/
│   ├── preprocess.py                # PySpark cleaning and validation
│   └── build_features.py            # PySpark feature engineering
├── training/
│   ├── __init__.py
│   ├── model.py                     # PyTorch neural network
│   └── train.py                     # Training and evaluation
├── evaluation/
│   ├── __init__.py
│   └── diagnose_shift.py            # Distribution-shift analysis
├── tracking/
│   ├── __init__.py
│   └── log_current_run.py           # MLflow logging
├── models/
│   ├── metrics.json
│   └── train_test_distribution_shift.csv
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

### Requirements

- Python 3.10 or later
- Java 17
- macOS, Linux, or Windows with a supported Spark environment

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Verify the environment

```bash
python --version
java -version
python -c "import pyspark, torch, mlflow; print(pyspark.__version__, torch.__version__, mlflow.__version__)"
```

## Run the Complete Pipeline

```bash
# 1. Clean the raw Kaggle datasets
python spark/preprocess.py

# 2. Build numeric model features
python spark/build_features.py

# 3. Train and evaluate the PyTorch model
python -m training.train

# 4. Diagnose external distribution shift
python -m evaluation.diagnose_shift

# 5. Start MLflow
mlflow server --host 127.0.0.1 --port 8080

# 6. In another terminal, log the trained model
python -m tracking.log_current_run
```

## Key Lessons

- High validation accuracy does not guarantee production performance.
- External datasets may have different feature distributions and feature-target relationships.
- Internal and external evaluation should be reported separately.
- Model monitoring should include threshold-independent metrics such as ROC AUC.
- MLflow makes experiments reproducible by connecting parameters, metrics, artifacts, and model versions.
- PySpark and Parquet provide a reusable foundation for scaling the preprocessing pipeline.

## Future Improvements

- Automatically log metrics during every training epoch
- Add early stopping
- Tune the decision threshold using validation data
- Add confusion-matrix and ROC-curve artifacts
- Add automated hyperparameter search
- Add a FastAPI prediction endpoint
- Package the pipeline with Docker
- Add unit tests and GitHub Actions
- Add data-drift monitoring for new customer batches

## Resume Bullet

> Built an end-to-end customer churn prediction pipeline using PySpark for scalable ETL and feature engineering, PyTorch for neural-network classification, and MLflow for experiment tracking, artifact management, distribution-shift analysis, and model versioning.

## License

This project is intended for educational and portfolio use.
