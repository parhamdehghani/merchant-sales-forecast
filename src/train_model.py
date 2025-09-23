import os
import shutil
from src.spark_utils import get_or_create_spark_session
from src.data_processing import load_data, preprocess_transactions, prepare_for_forecasting
from src.forecasting_model import ForecastingModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE_PATH = os.path.join(BASE_DIR, "..", "monthly_transactions.csv") 
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "forecasting_pipeline_model") 

    # Ensure the model save directory exists and is empty
    if os.path.exists(MODEL_SAVE_PATH):
        import shutil
        shutil.rmtree(MODEL_SAVE_PATH)
        print(f"Removed existing model directory: {MODEL_SAVE_PATH}")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # 1. Initialize Spark Session
    spark = get_or_create_spark_session()

    try:
        # 2. Load and Preprocess Data
        print(f"Loading data from {DATA_FILE_PATH}...")
        raw_df = load_data(spark, DATA_FILE_PATH)
        preprocessed_df = preprocess_transactions(raw_df)
        historical_data_for_forecasting = prepare_for_forecasting(spark, preprocessed_df)
        historical_data_for_forecasting.cache() # Cache for training
        print("Data loaded and preprocessed for training.")

        # 3. Perform Time-based Train-Test Split (70% train, 30% test)
        print("Performing time-based train-test split...")
        
        # Get all unique months and sort them
        all_unique_months = historical_data_for_forecasting.select("transaction_month").distinct().orderBy("transaction_month").collect()
        num_total_months = len(all_unique_months)
        num_test_months = int(num_total_months * 0.30) # 30% for test set
        
        # Determine the cutoff month
        # If num_test_months is 0, this logic needs to be robust (e.g., if data is too small)
        if num_test_months == 0 and num_total_months > 0:
            # Ensure at least one month in test set if data exists
            test_cutoff_month = all_unique_months[-1].transaction_month
        elif num_test_months == 0 and num_total_months == 0:
            raise ValueError("No data available for train-test split.")
        else:
            test_cutoff_month_index = num_total_months - num_test_months
            test_cutoff_month = all_unique_months[test_cutoff_month_index].transaction_month
        
        training_df = historical_data_for_forecasting.filter(col("transaction_month") < test_cutoff_month)
        test_df = historical_data_for_forecasting.filter(col("transaction_month") >= test_cutoff_month)
        
        print(f"Training data contains {training_df.count()} records up to {test_cutoff_month.strftime('%Y%m')}")
        print(f"Test data contains {test_df.count()} records from {test_cutoff_month.strftime('%Y%m')} onwards")
        
        # 4. Initialize and Train Forecasting Model
        print("Training forecasting model...")
        forecasting_model_instance = ForecastingModel()
        forecasting_model_instance.train(spark, training_df) # Train on training_df
        print("Forecasting model training complete.")

        # 5. Save the trained model
        print(f"Saving trained model to {MODEL_SAVE_PATH}...")
        forecasting_model_instance.save(spark, MODEL_SAVE_PATH)
        print("Model saved successfully.")

        # 6. Evaluate the trained model on the TEST set
        print("Evaluating the trained model on the test set...")
        loaded_model_instance = ForecastingModel.load(spark, MODEL_SAVE_PATH)
        
        # Generate features on the test data using the loaded model's feature generation logic
        df_with_features_for_eval = loaded_model_instance._generate_features(spark, test_df) # Evaluate on test_df
        
        # Make predictions using the loaded pipeline model
        predictions = loaded_model_instance.model.transform(df_with_features_for_eval)
        
        # Evaluate predictions
        evaluator = RegressionEvaluator(labelCol="sales_amount_actual", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print(f"Model evaluation complete. RMSE on test data: {rmse}")

    except Exception as e:
        print(f"An error occurred during model training: {e}")
    finally:
        # Stop Spark Session
        spark.stop()
        print("SparkSession stopped.")
