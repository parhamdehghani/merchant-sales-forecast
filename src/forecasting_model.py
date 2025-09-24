from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, max, to_date, date_add, last_day, desc, row_number, add_months, date_trunc, lag, avg, month, year, sequence, explode, when, expr, rand
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.window import Window
from datetime import datetime
import os 
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator


class ForecastingModel:
    """
    A class to encapsulate the PySpark-based sales forecasting model.
    This model will be a single global model capable of forecasting for all merchants.
    """

    def __init__(self):
        """
        Initializes the ForecastingModel.
        """
        # self.spark = spark # Removed SparkSession from instance attributes
        self.model = None # This will be a PipelineModel after training/loading
        self.feature_cols = [f"sales_lag_{i}" for i in range(1, 4)] + \
                            ["ma_3_months", "ma_6_months", "month_of_year", "year"] + \
                            ["currency_code_encoded", "country_code_encoded"]
        self.pipeline = None

    def _generate_features(self, spark: SparkSession, df):
        """
        Generates time-series features for the forecasting model.
        This will include lagged sales, moving averages, and time-based features.

        Args:
            df (DataFrame): The preprocessed DataFrame with continuous monthly sales data.

        Returns:
            DataFrame: The DataFrame with generated features.
        """
        # This is a critical step for a single global model.
        # Features will include:
        # 1. Lagged sales amounts for previous months (t-1, t-2, t-3).
        # 2. Moving averages (3-month, 6-month).
        # 3. Time-based features: month of year, year.
        # 4. Merchant-specific historical aggregates (merchant's overall average sales).

        window_spec_merchant = Window.partitionBy("anonymous_uu_id").orderBy("transaction_month")

        # Lagged sales features
        df_with_features = df.withColumn("sales_lag_1", lag("sales_amount_actual", 1).over(window_spec_merchant))
        df_with_features = df_with_features.withColumn("sales_lag_2", lag("sales_amount_actual", 2).over(window_spec_merchant))
        df_with_features = df_with_features.withColumn("sales_lag_3", lag("sales_amount_actual", 3).over(window_spec_merchant))

        # Moving average features
        df_with_features = df_with_features.withColumn("ma_3_months", avg("sales_amount_actual").over(window_spec_merchant.rowsBetween(-2, 0)))
        df_with_features = df_with_features.withColumn("ma_6_months", avg("sales_amount_actual").over(window_spec_merchant.rowsBetween(-5, 0)))

        # Time-based features
        df_with_features = df_with_features.withColumn("month_of_year", month(col("transaction_month")))
        df_with_features = df_with_features.withColumn("year", year(col("transaction_month")))

        # Drop rows with nulls introduced by lagging 
        # For training, we only need rows where all features are present
        df_with_features = df_with_features.dropna(subset=self.feature_cols)

        return df_with_features

    def train(self, spark: SparkSession, historical_data_df):
        """
        Trains the global forecasting model using historical sales data.

        Args:
            spark (SparkSession): The active SparkSession.
            historical_data_df (DataFrame): Preprocessed DataFrame from data_processing.prepare_for_forecasting.
        """
        # 1. Generate features from historical data
        df_with_features = self._generate_features(spark, historical_data_df)
        
        # 2. Assemble features into a vector
        vector_assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features")

        # 3. Define the regression model using xgboost regressor
        xgb = SparkXGBRegressor(features_col="features", label_col="sales_amount_actual", 
                                objective="reg:squarederror", 
                                num_workers=4, 
                                seed=42)

        # 4. Create a Spark ML Pipeline
        pipeline = Pipeline(stages=[vector_assembler, xgb]) 

        # 5. Define ParamGrid for hyperparameter tuning
        param_grid = ParamGridBuilder() \
            .addGrid(xgb.max_depth, [5, 10]) \
            .addGrid(xgb.n_estimators, [300, 500]) \
            .build()

        # 6. Define an evaluator
        # We'll use RMSE as metric for regression.
        evaluator = RegressionEvaluator(labelCol="sales_amount_actual", predictionCol="prediction", metricName="rmse")

        # 7. Set up CrossValidator
        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=param_grid,
                                  evaluator=evaluator,
                                  numFolds=3, # 3-fold cross-validation
                                  seed=42)

        # 8. Run cross-validation and select the best model
        cv_model = crossval.fit(df_with_features)
        self.model = cv_model.bestModel
        
        print("Forecasting model trained successfully with cross-validation and hyperparameter tuning.")

    def save(self, spark: SparkSession, path: str):
        """
        Saves the trained PySpark PipelineModel to the specified path.

        Args:
            spark (SparkSession): The active SparkSession.
            path (str): The directory path to save the model.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Cannot save.")
        self.model.save(path)
        print(f"Forecasting model saved to {path}")

    @classmethod
    def load(cls, spark: SparkSession, path: str):
        """
        Loads a pre-trained PySpark PipelineModel from the specified path.
        
        Args:
            spark (SparkSession): The active SparkSession.
            path (str): The directory path from which to load the model.
        
        Returns:
            ForecastingModel: A new instance of ForecastingModel with the loaded model.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path not found: {path}")
        
        loaded_pipeline_model = PipelineModel.load(path)
        
        # Reconstruct the ForecastingModel instance
        instance = cls()
        instance.model = loaded_pipeline_model
        
        # Re-extract feature_cols from the loaded pipeline if possible or store them with the model
        # For simplicity, assuming feature_cols remain consistent for a given model version.
        for stage in loaded_pipeline_model.stages:
            if isinstance(stage, VectorAssembler):
                instance.feature_cols = stage.getInputCols()
                break

        print(f"Forecasting model loaded from {path}")
        return instance

    def predict(self, spark: SparkSession, historical_df, months_to_forecast: int = 6):
        """
        Predicts the monthly sales revenue for a specific merchant for the next `months_to_forecast`.
        This method now uses a more PySpark-idiomatic iterative forecasting approach
        to avoid costly `collect()` and `createDataFrame()` operations in a loop.

        Args:
            spark (SparkSession): The active SparkSession.
            historical_df (DataFrame): The historical sales data for the merchant(s) to forecast for,
                                           preprocessed by `data_processing.prepare_for_forecasting`.
            months_to_forecast (int): The number of months to forecast into the future.

        Returns:
            DataFrame: A DataFrame with predicted sales amounts for the next `months_to_forecast`.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet. Call train() or load() first.")

        # Ensure the historical data is ordered by month
        historical_df = historical_df.orderBy("anonymous_uu_id", "transaction_month")

        # Get the latest transaction month from the historical data for each merchant
        latest_month_per_merchant = historical_df.groupBy("anonymous_uu_id") \
                                                      .agg(max("transaction_month").alias("latest_month"))

        # Generate future months for each merchant
        future_months_df = latest_month_per_merchant.withColumn(
            "future_month_sequence",
            sequence(add_months(col("latest_month"), 1), add_months(col("latest_month"), months_to_forecast), expr("INTERVAL '1 month'"))
        ).withColumn("transaction_month", explode(col("future_month_sequence"))) \
         .select(
             col("anonymous_uu_id"),
             col("transaction_month"),
             lit(None).cast("double").alias("sales_amount_actual"), # Placeholder
             lit(None).cast("string").alias("currency_code"), # Placeholder
             lit(None).cast("double").alias("currency_code_indexed"), # Placeholder
             lit(None).cast(VectorUDT()).alias("currency_code_encoded"), # Placeholder
             lit(None).cast("string").alias("country_code"), # Placeholder
             lit(None).cast("double").alias("country_code_indexed"), # Placeholder
             lit(None).cast(VectorUDT()).alias("country_code_encoded") # Placeholder
         )

        # Combine historical data and future months for all merchants
        # This unified DataFrame will be iteratively updated with predictions
        combined_df = historical_df.unionByName(future_months_df)
        combined_df = combined_df.orderBy("anonymous_uu_id", "transaction_month")

        # Iteratively generate features and predict
        current_df_for_prediction = combined_df

        for i in range(1, months_to_forecast + 1):
            # Generate features based on the current state of combined_df (historical + previous predictions)
            df_with_features = self._generate_features(spark, current_df_for_prediction)

            # Filter for the month we are currently predicting
            month_to_predict_df = latest_month_per_merchant.withColumn("target_month", add_months(col("latest_month"), i))

            prediction_input_df = df_with_features.join(
                month_to_predict_df,
                (df_with_features.anonymous_uu_id == month_to_predict_df.anonymous_uu_id) &
                (df_with_features.transaction_month == month_to_predict_df.target_month),
                "inner"
            ).select(df_with_features["*"])

            # Apply the trained model to get predictions for this month
            predicted_df = self.model.transform(prediction_input_df)
            
            # Clamp predictions to non-negative values
            predictions_for_this_month = predicted_df.withColumn(
                "prediction", when(col("prediction") < 0, 0.0).otherwise(col("prediction"))
            )
            
            # Update current_df_for_prediction with the new predictions
            current_df_for_prediction = current_df_for_prediction.alias("c").join(
                predictions_for_this_month.select(
                    col("anonymous_uu_id"), col("transaction_month"), col("prediction").alias("new_sales")
                ).alias("p"),
                on=["anonymous_uu_id", "transaction_month"],
                how="left"
            ).withColumn(
                "sales_amount_actual",
                when(col("c.sales_amount_actual").isNull(), col("p.new_sales"))
                .otherwise(col("c.sales_amount_actual"))
            ).select("c.*", "sales_amount_actual")

        # Filter for the future months that were predicted
        final_predictions_df = current_df_for_prediction.join(
            future_months_df.select("anonymous_uu_id", "transaction_month").alias("f"),
            on=["anonymous_uu_id", "transaction_month"]
        ).select("f.anonymous_uu_id", "f.transaction_month", col("sales_amount_actual").alias("prediction"))

        return final_predictions_df.orderBy("transaction_month")
    
    def predict_fast(self, spark: SparkSession, historical_df, months_to_forecast: int = 6):
        """
        Ultra-fast prediction method for production API.
        Uses simple average of recent sales without complex feature engineering.
        """
        # Get average sales for the merchant from recent months
        recent_sales = historical_df.groupBy("anonymous_uu_id").agg(
            avg("sales_amount_actual").alias("avg_sales")
        )
        
        # Get latest month for each merchant
        latest_month = historical_df.groupBy("anonymous_uu_id").agg(
            max("transaction_month").alias("latest_month")
        )
        
        # Create future months
        future_months = latest_month.select(
            col("anonymous_uu_id"),
            explode(
                sequence(
                    add_months(col("latest_month"), 1),
                    add_months(col("latest_month"), months_to_forecast),
                    expr("INTERVAL '1 month'")
                )
            ).alias("transaction_month")
        )
        
        # Join with average sales and add some variation
        predictions = future_months.join(recent_sales, on="anonymous_uu_id").select(
            col("anonymous_uu_id"),
            col("transaction_month"),
            # Add small random variation to average sales
            (col("avg_sales") * (1.0 + (rand() - 0.5) * 0.1)).alias("prediction")
        )
        
        return predictions.select(
            col("anonymous_uu_id"),
            col("transaction_month"),
            when(col("prediction") < 0, 0.0).otherwise(col("prediction")).alias("prediction")
        ).orderBy("anonymous_uu_id", "transaction_month")
