import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType
from datetime import datetime
from pyspark.sql.functions import col

from src.data_processing import load_data, preprocess_transactions, prepare_for_forecasting, generate_eligibility_features

@pytest.fixture(scope="module")
def spark_session():
    """
    Fixture for creating a SparkSession for testing.
    """
    spark = SparkSession.builder \
        .appName("TestDataProcessing") \
        .master("local[*]") \
        .getOrCreate()
    yield spark
    spark.stop()

def test_load_data(spark_session, tmp_path):
    """
    Tests the load_data function.
    """
    # Create a dummy CSV file
    csv_content = """
anonymous_uu_id,currency_code,country_code,transaction_month,sales_amount,transaction_count
merchant1,USD,US,202301,100.50,10
merchant2,EUR,FR,202301,200.75,20
merchant1,USD,US,202302,150.00,15
"""
    data_file = tmp_path / "monthly_transactions.csv"
    data_file.write_text(csv_content)

    df = load_data(spark_session, str(data_file))

    assert df.count() == 3
    assert "anonymous_uu_id" in df.columns
    assert "sales_amount" in df.columns
    assert df.schema["sales_amount"].dataType == DoubleType() # Inferred as double with inferSchema=True

def test_preprocess_transactions(spark_session):
    """
    Tests the preprocess_transactions function.
    """
    schema = StructType([
        StructField("anonymous_uu_id", StringType(), True),
        StructField("currency_code", StringType(), True),
        StructField("country_code", StringType(), True),
        StructField("transaction_month", StringType(), True),
        StructField("sales_amount", StringType(), True),
        StructField("transaction_count", StringType(), True)
    ])
    data = [
        ("m1", "USD", "US", "202301", "100.0", "10"),
        ("m1", "USD", "US", "202302", "200.5", "20"),
        ("m2", "EUR", "FR", "202301", "", "5"), # Missing sales amount
        ("m3", "GBP", "UK", "202303", "150.0", None) # Missing transaction count
    ]
    raw_df = spark_session.createDataFrame(data, schema)

    processed_df = preprocess_transactions(raw_df)

    assert processed_df.count() == 4
    assert "transaction_month_dt" in processed_df.columns
    assert processed_df.schema["transaction_month_dt"].dataType == DateType()
    assert processed_df.schema["sales_amount"].dataType == DoubleType()
    assert processed_df.schema["transaction_count"].dataType == LongType()

    # Check for new categorical feature columns
    assert "currency_code_indexed" in processed_df.columns
    assert "country_code_indexed" in processed_df.columns
    assert "currency_code_encoded" in processed_df.columns
    assert "country_code_encoded" in processed_df.columns

    # Check null handling
    assert processed_df.filter(col("anonymous_uu_id") == "m2").select("sales_amount").collect()[0][0] == 0.0
    assert processed_df.filter(col("anonymous_uu_id") == "m3").select("transaction_count").collect()[0][0] == 0

def test_prepare_for_forecasting(spark_session):
    """
    Tests the prepare_for_forecasting function.
    Ensures continuous time series and zero-filling for missing months.
    """
    schema = StructType([
        StructField("anonymous_uu_id", StringType(), True),
        StructField("currency_code", StringType(), True),
        StructField("country_code", StringType(), True),
        StructField("transaction_month_dt", DateType(), True),
        StructField("sales_amount", DoubleType(), True),
        StructField("transaction_count", LongType(), True)
    ])
    data = [
        ("m1", "USD", "US", datetime(2023, 1, 1), 100.0, 10),
        ("m1", "USD", "US", datetime(2023, 3, 1), 200.0, 20), # Missing Feb 2023
        ("m2", "EUR", "FR", datetime(2023, 1, 1), 50.0, 5)
    ]
    preprocessed_df = spark_session.createDataFrame(data, schema)

    prepared_df = prepare_for_forecasting(spark_session, preprocessed_df)

    # Expected months for m1: Jan, Feb, Mar 2023
    # Expected months for m2: Jan 2023
    assert prepared_df.filter(col("anonymous_uu_id") == "m1").count() == 3
    assert prepared_df.filter(col("anonymous_uu_id") == "m2").count() == 3 # m2 should also have data filled up to max date

    # Check for filled missing month
    m1_feb_sales = prepared_df.filter(
        (col("anonymous_uu_id") == "m1") & (col("transaction_month") == datetime(2023, 2, 1))
    ).select("sales_amount_actual").collect()[0][0]
    assert m1_feb_sales == 0.0

    # Check actual sales are preserved
    m1_jan_sales = prepared_df.filter(
        (col("anonymous_uu_id") == "m1") & (col("transaction_month") == datetime(2023, 1, 1))
    ).select("sales_amount_actual").collect()[0][0]
    assert m1_jan_sales == 100.0

def test_generate_eligibility_features(spark_session):
    """
    Tests the generate_eligibility_features function.
    Covers consecutive sales, average sales, and sales trend.
    """
    schema = StructType([
        StructField("anonymous_uu_id", StringType(), True),
        StructField("transaction_month_dt", DateType(), True),
        StructField("sales_amount_actual", DoubleType(), True)
    ])

    # Scenario 1: Merchant with sufficient history, good avg sales, positive trend
    data_m1 = [
        ("m1", datetime(2023, 1, 1), 5000.0),
        ("m1", datetime(2023, 2, 1), 5200.0),
        ("m1", datetime(2023, 3, 1), 5300.0),
        ("m1", datetime(2023, 4, 1), 5500.0),
        ("m1", datetime(2023, 5, 1), 5600.0),
        ("m1", datetime(2023, 6, 1), 5800.0),
        ("m1", datetime(2023, 7, 1), 6000.0),
        ("m1", datetime(2023, 8, 1), 6100.0),
        ("m1", datetime(2023, 9, 1), 6300.0),
        ("m1", datetime(2023, 10, 1), 6500.0),
        ("m1", datetime(2023, 11, 1), 6700.0),
        ("m1", datetime(2023, 12, 1), 7000.0), # 12 consecutive months, avg > 5000, positive trend
    ]

    # Scenario 2: Merchant with insufficient history (less than 12 months)
    data_m2 = [
        ("m2", datetime(2023, 10, 1), 4000.0),
        ("m2", datetime(2023, 11, 1), 4500.0),
        ("m2", datetime(2023, 12, 1), 5000.0), # 3 months history
    ]

    # Scenario 3: Merchant with low average sales
    data_m3 = [
        ("m3", datetime(2023, 1, 1), 1000.0),
        ("m3", datetime(2023, 2, 1), 1200.0),
        ("m3", datetime(2023, 3, 1), 1100.0),
        ("m3", datetime(2023, 4, 1), 1300.0),
        ("m3", datetime(2023, 5, 1), 1400.0),
        ("m3", datetime(2023, 6, 1), 1500.0),
        ("m3", datetime(2023, 7, 1), 1600.0),
        ("m3", datetime(2023, 8, 1), 1700.0),
        ("m3", datetime(2023, 9, 1), 1800.0),
        ("m3", datetime(2023, 10, 1), 1900.0),
        ("m3", datetime(2023, 11, 1), 2000.0),
        ("m3", datetime(2023, 12, 1), 2100.0), # 12 months, avg < 5000
    ]

    # Scenario 4: Merchant with negative trend (significant drop)
    data_m4 = [
        ("m4", datetime(2023, 1, 1), 10000.0),
        ("m4", datetime(2023, 2, 1), 9000.0),
        ("m4", datetime(2023, 3, 1), 8000.0),
        ("m4", datetime(2023, 4, 1), 7000.0),
        ("m4", datetime(2023, 5, 1), 6000.0),
        ("m4", datetime(2023, 6, 1), 5000.0),
        ("m4", datetime(2023, 7, 1), 2000.0),
        ("m4", datetime(2023, 8, 1), 1500.0),
        ("m4", datetime(2023, 9, 1), 1000.0),
        ("m4", datetime(2023, 10, 1), 900.0),
        ("m4", datetime(2023, 11, 1), 800.0),
        ("m4", datetime(2023, 12, 1), 700.0), # 12 months, avg > 5000, negative trend
    ]
    # Scenario 5: Merchant with a gap in sales (not consecutive)
    data_m5 = [
        ("m5", datetime(2023, 1, 1), 6000.0),
        ("m5", datetime(2023, 2, 1), 6200.0),
        ("m5", datetime(2023, 3, 1), 6500.0),
        ("m5", datetime(2023, 5, 1), 7000.0), # Gap in April
        ("m5", datetime(2023, 6, 1), 7200.0),
        ("m5", datetime(2023, 7, 1), 7500.0),
        ("m5", datetime(2023, 8, 1), 7800.0),
        ("m5", datetime(2023, 9, 1), 8000.0),
        ("m5", datetime(2023, 10, 1), 8200.0),
        ("m5", datetime(2023, 11, 1), 8500.0),
        ("m5", datetime(2023, 12, 1), 8800.0), # 11 months of sales, but not 12 consecutive.
    ]
    # Scenario 6: Merchant with zero sales in prior 3 months but positive in recent 3
    data_m6 = [
        ("m6", datetime(2023, 1, 1), 6000.0),
        ("m6", datetime(2023, 2, 1), 6000.0),
        ("m6", datetime(2023, 3, 1), 6000.0),
        ("m6", datetime(2023, 4, 1), 0.0),
        ("m6", datetime(2023, 5, 1), 0.0),
        ("m6", datetime(2023, 6, 1), 0.0),
        ("m6", datetime(2023, 7, 1), 500.0),
        ("m6", datetime(2023, 8, 1), 1000.0),
        ("m6", datetime(2023, 9, 1), 1500.0),
        ("m6", datetime(2023, 10, 1), 6000.0),
        ("m6", datetime(2023, 11, 1), 6500.0),
        ("m6", datetime(2023, 12, 1), 7000.0), # 12 months, avg > 5000, positive trend from zero
    ]

    full_data = data_m1 + data_m2 + data_m3 + data_m4 + data_m5 + data_m6
    df = spark_session.createDataFrame(full_data, schema)

    # First, ensure the data is prepared correctly for eligibility features (i.e., continuous series)
    prepared_df = prepare_for_forecasting(spark_session, df)

    # Calculate eligibility features
    eligibility_features_df = generate_eligibility_features(prepared_df)
    eligibility_features_df.persist()
    eligibility_features_df.show()

    # Assertions for Merchant 1: Eligible
    m1_features = eligibility_features_df.filter(col("anonymous_uu_id") == "m1").collect()[0]
    assert m1_features.consecutive_months_sales == 12
    assert m1_features.avg_sales_last_12_months > 5000.0
    assert m1_features.sales_trend_score > -0.1 # Allowing for slight dip
    assert m1_features.has_sufficient_history is True
    assert m1_features.meets_min_avg_sales is True
    assert m1_features.has_positive_trend is True

    # Assertions for Merchant 2: Not eligible (insufficient history)
    m2_features = eligibility_features_df.filter(col("anonymous_uu_id") == "m2").collect()[0]
    assert m2_features.consecutive_months_sales == 3
    assert m2_features.avg_sales_last_12_months == 1125.0 # (4000 + 4500 + 5000 + 9*0) / 12 = 1125.0
    assert m2_features.has_sufficient_history is False
    assert m2_features.meets_min_avg_sales is False # avg_sales_last_12_months < 5000
    assert m2_features.has_positive_trend is True # Trend based on 3 months of data, with prior 3 months being 0

    # Assertions for Merchant 3: Not eligible (low average sales)
    m3_features = eligibility_features_df.filter(col("anonymous_uu_id") == "m3").collect()[0]
    assert m3_features.consecutive_months_sales == 12
    assert m3_features.avg_sales_last_12_months < 5000.0
    assert m3_features.has_sufficient_history is True
    assert m3_features.meets_min_avg_sales is False
    assert m3_features.has_positive_trend is True # Trend should be positive based on small growth

    # Assertions for Merchant 4: Not eligible (negative trend)
    m4_features = eligibility_features_df.filter(col("anonymous_uu_id") == "m4").collect()[0]
    assert m4_features.consecutive_months_sales == 12
    assert m4_features.avg_sales_last_12_months == 4325.0 # Average of declining sales
    assert m4_features.sales_trend_score < -0.1 # Significant negative trend
    assert m4_features.has_sufficient_history is True
    assert m4_features.meets_min_avg_sales is False # avg < 5000
    assert m4_features.has_positive_trend is False

    # Assertions for Merchant 5: Not eligible (gap in sales, not 12 consecutive)
    m5_features = eligibility_features_df.filter(col("anonymous_uu_id") == "m5").collect()[0]
    assert m5_features.consecutive_months_sales == 8 # latest streak is from May to Dec
    assert m5_features.avg_sales_last_12_months > 5000.0
    assert m5_features.has_sufficient_history is False
    assert m5_features.meets_min_avg_sales is True
    assert m5_features.has_positive_trend is True

    # Assertions for Merchant 6: Not eligible (latest streak only 6 months)
    m6_features = eligibility_features_df.filter(col("anonymous_uu_id") == "m6").collect()[0]
    assert m6_features.consecutive_months_sales == 6 # latest streak is from Jul to Dec (6 months)
    assert m6_features.avg_sales_last_12_months == 3375.0 # Average over 12 months including zeros
    assert m6_features.sales_trend_score > 0.0 # Should be a very high positive trend
    assert m6_features.has_sufficient_history is False # because latest streak is 6 months (< 12)
    assert m6_features.meets_min_avg_sales is False # avg_sales_last_12_months < 5000
    assert m6_features.has_positive_trend is True
