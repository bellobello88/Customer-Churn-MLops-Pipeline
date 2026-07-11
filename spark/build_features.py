from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


TRAIN_INPUT_PATH = "data/processed/churn_train"
TEST_INPUT_PATH = "data/processed/churn_test"

TRAIN_OUTPUT_PATH = "data/processed/features_train"
TEST_OUTPUT_PATH = "data/processed/features_test"


def create_spark_session() -> SparkSession:
    """Create the local Spark session."""

    return (
        SparkSession.builder
        .master("local[*]")
        .appName("CustomerChurnFeatureEngineering")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def build_features(dataframe: DataFrame) -> DataFrame:
    """
    Convert the cleaned customer dataset into numeric features
    that can be used by a PyTorch neural network.
    """

    return dataframe.select(
        # Keep ID only for tracing predictions.
        F.col("customer_id").cast("long").alias("customer_id"),

        # Numeric features.
        F.col("age").cast("double").alias("age"),
        F.col("tenure").cast("double").alias("tenure"),
        F.col("usage_frequency")
        .cast("double")
        .alias("usage_frequency"),
        F.col("support_calls")
        .cast("double")
        .alias("support_calls"),
        F.col("payment_delay")
        .cast("double")
        .alias("payment_delay"),
        F.col("total_spend")
        .cast("double")
        .alias("total_spend"),
        F.col("last_interaction")
        .cast("double")
        .alias("last_interaction"),

        # Gender one-hot encoding.
        F.when(
            F.lower(F.col("gender")) == "male",
            1.0,
        ).otherwise(0.0).alias("gender_male"),

        F.when(
            F.lower(F.col("gender")) == "female",
            1.0,
        ).otherwise(0.0).alias("gender_female"),

        # Subscription type one-hot encoding.
        F.when(
            F.lower(F.col("subscription_type")) == "basic",
            1.0,
        ).otherwise(0.0).alias("subscription_basic"),

        F.when(
            F.lower(F.col("subscription_type")) == "standard",
            1.0,
        ).otherwise(0.0).alias("subscription_standard"),

        F.when(
            F.lower(F.col("subscription_type")) == "premium",
            1.0,
        ).otherwise(0.0).alias("subscription_premium"),

        # Contract length one-hot encoding.
        F.when(
            F.lower(F.col("contract_length")) == "monthly",
            1.0,
        ).otherwise(0.0).alias("contract_monthly"),

        F.when(
            F.lower(F.col("contract_length")) == "quarterly",
            1.0,
        ).otherwise(0.0).alias("contract_quarterly"),

        F.when(
            F.lower(F.col("contract_length")) == "annual",
            1.0,
        ).otherwise(0.0).alias("contract_annual"),

        # Target label.
        F.col("churn").cast("double").alias("churn"),
    )


def process_features(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    dataset_name: str,
) -> None:
    """Create and save numeric model features."""

    print("\n" + "=" * 60)
    print(f"Building features for {dataset_name} dataset")
    print("=" * 60)

    input_df = spark.read.parquet(input_path)

    print(f"Input rows: {input_df.count()}")

    features_df = build_features(input_df).cache()

    print(f"Feature rows: {features_df.count()}")
    print(f"Feature columns: {len(features_df.columns)}")

    print("\nFeature schema:")
    features_df.printSchema()

    print("\nFeature sample:")
    features_df.show(
        5,
        truncate=False,
    )

    print("\nChecking null values:")

    null_counts = features_df.select(
        [
            F.sum(
                F.col(column_name).isNull().cast("integer")
            ).alias(column_name)
            for column_name in features_df.columns
        ]
    )

    null_counts.show(
        truncate=False,
    )

    # Reduce the number of small Parquet files for local development.
    (
        features_df
        .coalesce(8)
        .write
        .mode("overwrite")
        .parquet(output_path)
    )

    verified_df = spark.read.parquet(output_path)

    print(f"Verified output rows: {verified_df.count()}")
    print(f"Saved feature data to: {output_path}")

    features_df.unpersist()


def main() -> None:
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        process_features(
            spark=spark,
            input_path=TRAIN_INPUT_PATH,
            output_path=TRAIN_OUTPUT_PATH,
            dataset_name="training",
        )

        process_features(
            spark=spark,
            input_path=TEST_INPUT_PATH,
            output_path=TEST_OUTPUT_PATH,
            dataset_name="testing",
        )

        print("\n" + "=" * 60)
        print("Feature engineering completed successfully.")
        print("=" * 60)

    except Exception as error:
        print(f"\nFeature engineering failed: {error}")
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()