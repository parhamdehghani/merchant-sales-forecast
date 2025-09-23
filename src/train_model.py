import os
import shutil
from src.spark_utils import get_or_create_spark_session
from src.data_processing import load_data, preprocess_transactions, prepare_for_forecasting
from src.forecasting_model import ForecastingModel

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

        # 3. Initialize and Train Forecasting Model
        print("Training forecasting model...")
        forecasting_model_instance = ForecastingModel()
        forecasting_model_instance.train(spark, historical_data_for_forecasting)
        print("Forecasting model training complete.")

        # 4. Save the trained model
        print(f"Saving trained model to {MODEL_SAVE_PATH}...")
        forecasting_model_instance.save(spark, MODEL_SAVE_PATH)
        print("Model saved successfully.")

    except Exception as e:
        print(f"An error occurred during model training: {e}")
    finally:
        # Stop Spark Session
        spark.stop()
        print("SparkSession stopped.")
