from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


TRAIN_INPUT_PATH = (
    "data/raw/customer_churn_dataset-training-master.csv"
)

TEST_INPUT_PATH = (
    "data/raw/customer_churn_dataset-testing-master.csv"
)

TRAIN_OUTPUT_PATH = "data/processed/churn_train"
TEST_OUTPUT_PATH = "data/processed/churn_test"


CUSTOMER_SCHEMA = T.StructType(
    [
        T.StructField("CustomerID", T.IntegerType(), True),
        T.StructField("Age", T.IntegerType(), True),
        T.StructField("Gender", T.StringType(), True),
        T.StructField("Tenure", T.IntegerType(), True),
        T.StructField("Usage Frequency", T.IntegerType(), True),
        T.StructField("Support Calls", T.IntegerType(), True),
        T.StructField("Payment Delay", T.IntegerType(), True),
        T.StructField("Subscription Type", T.StringType(), True),
        T.StructField("Contract Length", T.StringType(), True),
        T.StructField("Total Spend", T.DoubleType(), True),
        T.StructField("Last Interaction", T.IntegerType(), True),
        T.StructField("Churn", T.IntegerType(), True),
    ]
)


def create_spark_session() -> SparkSession:
    """Create a local Spark session."""

    return (
        SparkSession.builder
        .master("local[*]")
        .appName("CustomerChurnETL")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def read_customer_data(
    spark: SparkSession,
    input_path: str,
) -> DataFrame:
    """Read a customer churn CSV using an explicit schema."""

    return (
        spark.read
        .option("header", True)
        .schema(CUSTOMER_SCHEMA)
        .csv(input_path)
    )


def clean_customer_data(
    dataframe: DataFrame,
) -> DataFrame:
    """Rename, clean, and validate customer data."""

    cleaned_df = dataframe.select(
        F.col("CustomerID").alias("customer_id"),
        F.col("Age").alias("age"),
        F.trim(F.col("Gender")).alias("gender"),
        F.col("Tenure").alias("tenure"),
        F.col("Usage Frequency").alias("usage_frequency"),
        F.col("Support Calls").alias("support_calls"),
        F.col("Payment Delay").alias("payment_delay"),
        F.trim(
            F.col("Subscription Type")
        ).alias("subscription_type"),
        F.trim(
            F.col("Contract Length")
        ).alias("contract_length"),
        F.col("Total Spend").alias("total_spend"),
        F.col("Last Interaction").alias("last_interaction"),
        F.col("Churn").alias("churn"),
    )

    # Remove rows without an ID or target label.
    cleaned_df = cleaned_df.dropna(
        subset=["customer_id", "churn"]
    )

    # Fill missing feature values.
    cleaned_df = cleaned_df.fillna(
        {
            "age": 0,
            "gender": "Unknown",
            "tenure": 0,
            "usage_frequency": 0,
            "support_calls": 0,
            "payment_delay": 0,
            "subscription_type": "Unknown",
            "contract_length": "Unknown",
            "total_spend": 0.0,
            "last_interaction": 0,
        }
    )

    # Remove duplicated customer IDs.
    cleaned_df = cleaned_df.dropDuplicates(
        ["customer_id"]
    )

    # Keep only binary churn labels.
    cleaned_df = cleaned_df.filter(
        F.col("churn").isin(0, 1)
    )

    # Remove invalid negative numeric values.
    cleaned_df = cleaned_df.filter(
        (F.col("age") >= 0)
        & (F.col("tenure") >= 0)
        & (F.col("usage_frequency") >= 0)
        & (F.col("support_calls") >= 0)
        & (F.col("payment_delay") >= 0)
        & (F.col("total_spend") >= 0)
        & (F.col("last_interaction") >= 0)
    )

    return cleaned_df


def process_dataset(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    dataset_name: str,
) -> None:
    """Process and save one customer churn dataset."""

    print("\n" + "=" * 60)
    print(f"Processing {dataset_name} dataset")
    print("=" * 60)

    raw_df = read_customer_data(
        spark=spark,
        input_path=input_path,
    )

    raw_count = raw_df.count()

    print(f"Input path: {input_path}")
    print(f"Raw row count: {raw_count}")

    cleaned_df = clean_customer_data(raw_df).cache()

    cleaned_count = cleaned_df.count()

    print(f"Cleaned row count: {cleaned_count}")
    print(f"Removed row count: {raw_count - cleaned_count}")

    print("\nCleaned schema:")
    cleaned_df.printSchema()

    print("\nSample customer records:")
    cleaned_df.show(
        5,
        truncate=False,
    )

    print("\nChurn distribution:")

    (
        cleaned_df
        .groupBy("churn")
        .count()
        .orderBy("churn")
        .show()
    )

    print("\nCategorical values:")

    (
        cleaned_df
        .groupBy("subscription_type")
        .count()
        .orderBy("subscription_type")
        .show()
    )

    (
        cleaned_df.write
        .mode("overwrite")
        .parquet(output_path)
    )

    verified_df = spark.read.parquet(output_path)

    print(f"Verified Parquet rows: {verified_df.count()}")
    print(f"Saved Parquet data to: {output_path}")

    cleaned_df.unpersist()


def main() -> None:
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        process_dataset(
            spark=spark,
            input_path=TRAIN_INPUT_PATH,
            output_path=TRAIN_OUTPUT_PATH,
            dataset_name="training",
        )

        process_dataset(
            spark=spark,
            input_path=TEST_INPUT_PATH,
            output_path=TEST_OUTPUT_PATH,
            dataset_name="testing",
        )

        print("\n" + "=" * 60)
        print("Customer churn ETL completed successfully.")
        print("=" * 60)

    except Exception as error:
        print(f"\nETL failed: {error}")
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()