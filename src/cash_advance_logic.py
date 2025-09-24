from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, avg, count, when, datediff, last_day, max, lit, date_sub, lag, row_number, add_months, expr, desc
from pyspark.sql.window import Window
from datetime import datetime

class CashAdvanceCalculator:
    """
    A class to determine merchant eligibility for cash advances and calculate advance amounts.
    """
    FEE = 0.15  # 15% flat fee
    HOLDBACK = 0.10  # 10% holdback percentage
    MIN_AVG_MONTHLY_SALES_THRESHOLD = 5000.0 # This threshold can be changed
    REQUIRED_HISTORY_MONTHS = 12 # Minimum consecutive months with sales

    def __init__(self, spark: SparkSession):
        """
        Initializes the CashAdvanceCalculator.

        Args:
            spark (SparkSession): The active SparkSession.
        """
        self.spark = spark

    def _calculate_historical_features(self, df: DataFrame, as_of_date: datetime):
        """
        Calculates historical features for eligibility assessment for all merchants
        up to a specified `as_of_date`.

        Args:
            df (DataFrame): The preprocessed DataFrame with continuous monthly sales data
                            (`anonymous_uu_id`, `transaction_month`, `sales_amount_actual`).
            as_of_date (datetime): The date up to which historical features should be calculated.
                                   Here, it is the last month of available historical data.

        Returns:
            DataFrame: A DataFrame with calculated historical features for each merchant.
        """
        # Filter data up to the as_of_date
        historical_data = df.filter(col("transaction_month") <= as_of_date)

        window_spec_merchant = Window.partitionBy("anonymous_uu_id").orderBy("transaction_month")
        window_spec_merchant_unbounded = Window.partitionBy("anonymous_uu_id").orderBy("transaction_month").rowsBetween(Window.unboundedPreceding, Window.currentRow)

        # 1. Consecutive Months with Sales (> 0)
        # We want to find the longest streak of consecutive months with sales > 0
        # ending at or before `as_of_date`.
        # Filter to months with positive sales
        positive_sales_history = historical_data.filter(col("sales_amount_actual") > 0)

        # Assign a row number to each positive sales month for each merchant
        window_spec_rn = Window.partitionBy("anonymous_uu_id").orderBy("transaction_month")
        positive_sales_with_rn = positive_sales_history.withColumn(
            "row_num", row_number().over(window_spec_rn)
        )

        # Calculate the difference between `transaction_month` and `row_num` (in months).
        # This creates a constant `streak_identifier` for consecutive months.
        # We use a custom expression for date_sub_months as PySpark's datediff is in days.
        # Convert transaction_month to total months since epoch for subtraction
        positive_sales_with_streak_id = positive_sales_with_rn.withColumn(
            "streak_identifier",
            expr("months_between(transaction_month, '1970-01-01')").cast("int") - col("row_num")
        )

        # Group by merchant and streak_identifier to count consecutive months
        consecutive_months_df = positive_sales_with_streak_id.groupBy(
            "anonymous_uu_id", "streak_identifier"
        ).agg(
            count("transaction_month").alias("consecutive_count"),
            max("transaction_month").alias("streak_end_date")
        )

        # Get the latest (by `streak_end_date`) consecutive count for each merchant
        latest_consecutive_counts = consecutive_months_df.withColumn(
            "rn", row_number().over(Window.partitionBy("anonymous_uu_id").orderBy(desc("streak_end_date")))
        ).filter(col("rn") == 1)

        # Select the final consecutive months count, handling cases with no positive sales
        latest_consecutive_months_with_sales = latest_consecutive_counts.select(
            col("anonymous_uu_id"),
            col("consecutive_count").alias("latest_consecutive_months_with_sales")
        )
        # If a merchant has no positive sales history, this join will result in null, which fillna will handle.

        # 2. Average Monthly Sales (Last 12 Months)
        # Determine the start date for the last 12 months using add_months for precision
        twelve_months_ago_dt = add_months(lit(as_of_date), -12)

        last_12_months_sales = historical_data.filter(
            (col("transaction_month") >= twelve_months_ago_dt) & (col("transaction_month") <= lit(as_of_date))
        ).groupBy("anonymous_uu_id").agg(
            avg("sales_amount_actual").alias("avg_sales_last_12_months"),
            count(when(col("sales_amount_actual") > 0, True)).alias("months_in_last_12_with_sales") # Count months with positive sales
        )

        # 3. Sales Trend (Last 6 Months: Recent 3 vs. Prior 3)
        # Define precise month boundaries for 3-month segments
        six_months_ago_dt = add_months(lit(as_of_date), -6)
        three_months_ago_dt = add_months(lit(as_of_date), -3)

        recent_3_months_sales = historical_data.filter(
            (col("transaction_month") >= three_months_ago_dt) & (col("transaction_month") <= lit(as_of_date))
        ).groupBy("anonymous_uu_id").agg(
            avg("sales_amount_actual").alias("avg_sales_recent_3_months")
        )

        prior_3_months_sales = historical_data.filter(
            (col("transaction_month") >= six_months_ago_dt) & (col("transaction_month") < three_months_ago_dt) # Note '<' for prior 3 months
        ).groupBy("anonymous_uu_id").agg(
            avg("sales_amount_actual").alias("avg_sales_prior_3_months")
        )

        # Join all features
        features_df = last_12_months_sales.join(recent_3_months_sales, "anonymous_uu_id", "left_outer") \
                                          .join(prior_3_months_sales, "anonymous_uu_id", "left_outer") \
                                          .join(latest_consecutive_months_with_sales, "anonymous_uu_id", "left_outer")

        # Fill NaNs for averages and counts that might arise from missing historical data
        features_df = features_df.fillna(0, subset=["avg_sales_last_12_months", "avg_sales_recent_3_months",
                                                   "avg_sales_prior_3_months", "latest_consecutive_months_with_sales",
                                                   "months_in_last_12_with_sales"])

        return features_df

    def determine_eligibility(self, historical_features_df: DataFrame) -> DataFrame:
        """
        Determines the eligibility of merchants for a cash advance based on calculated historical features.

        Args:
            historical_features_df (DataFrame): A DataFrame containing historical features for each merchant,
                                                output from `_calculate_historical_features`.

        Returns:
            DataFrame: A DataFrame with 'anonymous_uu_id' and a boolean 'is_eligible' column.
        """
        # Apply eligibility criteria
        eligible_merchants_df = historical_features_df \
            .withColumn("is_sufficient_history",
                        (col("latest_consecutive_months_with_sales") >= self.REQUIRED_HISTORY_MONTHS))

        eligible_merchants_df = eligible_merchants_df \
            .withColumn("meets_min_avg_sales",
                        (col("avg_sales_last_12_months") >= self.MIN_AVG_MONTHLY_SALES_THRESHOLD))

        # Sales trend: (avg_sales_recent_3_months >= avg_sales_prior_3_months * 0.95) - allowing for a maximum 5% drop
        # Handle cases where prior_3_months_sales might be 0 to avoid division by zero or incorrect trend assessment.
        # If prior_3_months_sales is 0, a positive recent_3_months_sales indicates a positive trend.
        # Otherwise, check for stability (at most a 10% drop) or positive growth.
        eligible_merchants_df = eligible_merchants_df \
            .withColumn("has_stable_or_positive_trend",
                        when(col("avg_sales_prior_3_months") == 0,
                             col("avg_sales_recent_3_months") > 0) # If prior is 0, any recent sales is a positive trend
                        .otherwise(
                             (col("avg_sales_recent_3_months") >= (col("avg_sales_prior_3_months") * 0.95)) # Allow for up to 5% drop
                        ))

        eligible_merchants_df = eligible_merchants_df \
            .withColumn("is_eligible",
                        (col("is_sufficient_history") & col("meets_min_avg_sales") & col("has_stable_or_positive_trend")))

        return eligible_merchants_df.select("anonymous_uu_id", "is_eligible")

    def calculate_advance_amount(self, merchant_id: str, forecasted_sales: list) -> float:
        """
        Calculates the cash advance amount for an eligible merchant.

        Args:
            merchant_id (str): The ID of the merchant.
            forecasted_sales (list): A list of 6-month forecasted sales amounts for the merchant.

        Returns:
            float: The calculated cash advance amount.
        """
        if not forecasted_sales:
            return 0.0

        # Volume of Sales = sum of forecasted sales for the 6-month repayment period
        volume_of_sales = sum(forecasted_sales)

        # Offer Size = Volume of Sales * Holdback / (1 + Fee)
        offer_size = volume_of_sales * self.HOLDBACK / (1 + self.FEE)

        return offer_size
