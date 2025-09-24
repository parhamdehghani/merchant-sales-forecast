from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, year, month, to_date, desc, when, lag, row_number, avg, min, max, sequence, explode, date_trunc, lit, last_day, add_months, expr
from pyspark.sql.window import Window
from pyspark.sql.types import DateType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler 

def load_data(spark, file_path):
    """
    Loads the monthly_transactions.csv dataset into a PySpark DataFrame.

    Args:
        spark (SparkSession): The active SparkSession.
        file_path (str): The path to the monthly_transactions.csv file.

    Returns:
        DataFrame: A PySpark DataFrame containing the raw transaction data.
    """
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df

def preprocess_transactions(df):
    """
    Preprocesses the raw transaction data.
    This includes:
    - Casting 'transaction_month' to date type.
    - Ensuring 'sales_amount' is numeric and handling potential NaNs (filling with 0 for sales).
    - Aggregating sales by merchant and month, if necessary (though the data seems pre-aggregated).
    - Handling categorical variables: currency_code and country_code.

    Args:
        df (DataFrame): The raw transaction DataFrame.

    Returns:
        DataFrame: The preprocessed DataFrame.
    """
    # Convert transaction_month to a proper date format
    # Assuming 'transaction_month' is in 'yyyyMM' format,
    # and we want to ensure it's treated as a month start for aggregation.
    df = df.withColumn("transaction_month_dt", to_date(col("transaction_month"), "yyyyMM"))

    # Handle potential nulls in sales_amount by filling with 0, and ensure it's a numeric type
    df = df.withColumn("sales_amount", col("sales_amount").cast("double"))
    df = df.fillna(0, subset=["sales_amount"])

    # Ensure transaction_count is numeric
    df = df.withColumn("transaction_count", col("transaction_count").cast("long"))
    df = df.fillna(0, subset=["transaction_count"])

    # For forecasting, we need total monthly sales per merchant.
    # The input `monthly_transactions.csv` is already at this granularity.
    # We will sort and ensure completeness for time series.
    df = df.orderBy("anonymous_uu_id", "transaction_month_dt")

    # Handle categorical variables: currency_code and country_code
    categorical_cols = ["currency_code", "country_code"]
    indexed_cols = [f"{c}_indexed" for c in categorical_cols]
    encoded_cols = [f"{c}_encoded" for c in categorical_cols]

    # StringIndexer to convert categorical strings to numerical indices
    string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=indexed_cols, handleInvalid="keep")
    model_string_indexer = string_indexer.fit(df)
    df = model_string_indexer.transform(df)

    # OneHotEncoder to convert indexed numerical values to one-hot encoded vectors
    # dropLast=True to avoid multicollinearity (N-1 categories)
    one_hot_encoder = OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols, dropLast=True)
    model_one_hot_encoder = one_hot_encoder.fit(df)
    df = model_one_hot_encoder.transform(df)

    return df

def prepare_for_forecasting(spark, df):
    """
    Prepares the data for time-series forecasting.
    This function ensures that for each merchant, we have a continuous time series
    of monthly sales, filling in any missing months with zero sales.

    Args:
        spark (SparkSession): The active SparkSession.
        df (DataFrame): The preprocessed transaction DataFrame.

    Returns:
        DataFrame: A DataFrame ready for forecasting, with a continuous monthly series
                   for each merchant.
    """
    # This step is critical for time-series forecasting to ensure no gaps.
    # Get min and max dates across all data to generate a full date range
    min_date_overall, max_date_overall = df.agg(
        date_trunc("month", min("transaction_month_dt")).alias("min_month"),
        date_trunc("month", max("transaction_month_dt")).alias("max_month")
    ).head()

    # Create a DataFrame with all months in the range
    # Using F.sequence directly to create date range
    all_months_df = spark.range(1).select(
        sequence(lit(min_date_overall), lit(max_date_overall), expr("INTERVAL '1 month'")).alias("all_month_dt")
    ).withColumn("all_month_dt", explode(col("all_month_dt")))
    all_months_df = all_months_df.withColumn("all_month", date_trunc("month", col("all_month_dt")))

    # Get all unique anonymous_uu_id
    all_uuids_df = df.select("anonymous_uu_id").distinct()

    # Cross join to get all possible (merchant_id, month) combinations
    full_series_df = all_uuids_df.crossJoin(all_months_df)

    # Left join with existing sales data
    # We join on merchant_id and the truncated month
    joined_df = full_series_df.join(
        df.withColumn("month_trunc", date_trunc("month", col("transaction_month_dt"))),
        (full_series_df.anonymous_uu_id == df.anonymous_uu_id) &
        (full_series_df.all_month == col("month_trunc")),
        "left_outer"
    )
    
    # Create select list based on available columns
    select_list = [
        full_series_df.anonymous_uu_id,
        full_series_df.all_month.alias("transaction_month") # Overwrite with transaction_month
    ]
    
    # Handle sales amount column (could be 'sales_amount' or 'sales_amount_actual')
    if "sales_amount_actual" in df.columns:
        select_list.append(col("sales_amount_actual"))
    elif "sales_amount" in df.columns:
        select_list.append(col("sales_amount").alias("sales_amount_actual"))
    else:
        # If neither exists, create a default column
        select_list.append(lit(0.0).alias("sales_amount_actual"))
    
    # Add categorical columns if they exist
    categorical_base_cols = ["currency_code", "country_code"]
    categorical_suffix_cols = ["_indexed", "_encoded"]
    
    for base_col in categorical_base_cols:
        if base_col in df.columns:
            select_list.append(col(base_col))
        else:
            select_list.append(lit(None).cast("string").alias(base_col))
            
        for suffix in categorical_suffix_cols:
            full_col = base_col + suffix
            if full_col in df.columns:
                select_list.append(col(full_col))
    
    joined_df = joined_df.select(*select_list).fillna(0, subset=["sales_amount_actual"]) # Fill missing months with 0 sales

    return joined_df.orderBy("anonymous_uu_id", "transaction_month") # Order by transaction_month

def generate_eligibility_features(df):
    """
    Generates features required for the cash advance eligibility logic.
    This includes:
    - Number of consecutive months with sales > 0.
    - Average monthly sales over the last 12 months (ending at the latest month for each merchant).
    - Sales trend over the last 6 months (comparing average of last 3 vs. prior 3 months).

    Args:
        df (DataFrame): The preprocessed DataFrame, after `prepare_for_forecasting`
                        to ensure continuous time series.

    Returns:
        DataFrame: A DataFrame with eligibility features for each merchant.
    """
    # Assuming df is the output of `prepare_for_forecasting`, meaning it has
    # `anonymous_uu_id`, `transaction_month`, and `sales_amount_actual`.

    window_spec_merchant = Window.partitionBy("anonymous_uu_id").orderBy("transaction_month")
    window_spec_merchant_unbounded = Window.partitionBy("anonymous_uu_id").orderBy("transaction_month").rowsBetween(Window.unboundedPreceding, Window.currentRow)

    # Identify streaks of positive sales
    df_with_streaks = df.withColumn(
        "is_positive_sales", when(col("sales_amount_actual") > 0, 1).otherwise(0)
    ).withColumn(
        "streak_start",
        when((col("is_positive_sales") == 1) & (lag("is_positive_sales", 1, 0).over(window_spec_merchant) == 0), 1)
        .otherwise(0)
    ).withColumn(
        "streak_id", sum("streak_start").over(window_spec_merchant_unbounded)
    )

    # Calculate consecutive months with sales > 0 within the latest streak
    window_spec_streak = Window.partitionBy("anonymous_uu_id", "streak_id").orderBy("transaction_month")
    df_with_streaks = df_with_streaks.withColumn(
        "consecutive_sales_months",
        when(col("is_positive_sales") == 1, count("is_positive_sales").over(window_spec_streak))
        .otherwise(0)
    )

    # Get the latest data point for each merchant to calculate eligibility features
    latest_data_df = df_with_streaks.withColumn("rn", row_number().over(Window.partitionBy("anonymous_uu_id").orderBy(desc("transaction_month"))))
    latest_data_df = latest_data_df.filter(col("rn") == 1).drop("rn")

    # Join back to get full historical data for window calculations
    df_joined_latest = df.alias("full_history").join(
        latest_data_df.alias("latest"),
        on="anonymous_uu_id",
        how="left"
    ).select(
        col("full_history.anonymous_uu_id"),
        col("full_history.transaction_month"), 
        col("full_history.sales_amount_actual"),
        col("latest.transaction_month").alias("latest_month_dt"), 
        col("latest.consecutive_sales_months").alias("latest_consecutive_sales_months"),
    )

    # Define dynamic windows relative to the latest_month_dt for each merchant
    window_12_months_preceding = Window.partitionBy("anonymous_uu_id").orderBy("transaction_month").rowsBetween(-11, 0)
    window_3_months_recent_preceding = Window.partitionBy("anonymous_uu_id").orderBy("transaction_month").rowsBetween(-2, 0)
    window_3_months_prior_preceding = Window.partitionBy("anonymous_uu_id").orderBy("transaction_month").rowsBetween(-5, -3)

    # Calculate average sales for the last 12 months and 6-month trend
    df_with_elig_features = df_joined_latest.withColumn(
        "avg_sales_last_12_months",
        avg("sales_amount_actual").over(window_12_months_preceding)
    ).withColumn(
        "avg_sales_last_3_months",
        avg("sales_amount_actual").over(window_3_months_recent_preceding)
    ).withColumn(
        "avg_sales_prior_3_months",
        avg("sales_amount_actual").over(window_3_months_prior_preceding)
    ).withColumn(
        "sales_trend_score", # (last 3 months avg - prior 3 months avg) / prior 3 months avg
        when(col("avg_sales_prior_3_months") > 0,
             (col("avg_sales_last_3_months") - col("avg_sales_prior_3_months")) / col("avg_sales_prior_3_months"))
        .otherwise(0.0) # Handle division by zero, assuming 0 trend if prior sales are 0
    )

    # Filter to get only the latest computed features for each merchant
    eligibility_features_df = df_with_elig_features.filter(col("transaction_month") == col("latest_month_dt")) \
                                                    .select(
                                                        "anonymous_uu_id",
                                                        col("latest_consecutive_sales_months").alias("consecutive_months_sales"),
                                                        "avg_sales_last_12_months",
                                                        "sales_trend_score"
                                                    )

    # Define eligibility flags based on criteria
    ELIGIBILITY_MIN_AVG_SALES = 5000.0 # We have ~ 11000 month+merchants with avg sales < 5000
    ELIGIBILITY_MIN_CONSECUTIVE_MONTHS = 12
    ELIGIBILITY_MIN_SALES_TREND_SCORE = -0.05 # Allow for slight negative trend, max 5% drop

    eligibility_features_df = eligibility_features_df \
        .withColumn("has_sufficient_history", col("consecutive_months_sales") >= ELIGIBILITY_MIN_CONSECUTIVE_MONTHS) \
        .withColumn("meets_min_avg_sales", col("avg_sales_last_12_months") >= ELIGIBILITY_MIN_AVG_SALES) \
        .withColumn("has_positive_trend", col("sales_trend_score") >= ELIGIBILITY_MIN_SALES_TREND_SCORE)

    return eligibility_features_df
