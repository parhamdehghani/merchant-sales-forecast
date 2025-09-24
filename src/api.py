from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
from datetime import datetime
from pyspark.sql.functions import col, max

from src.spark_utils import get_or_create_spark_session
from src.data_processing import load_data, preprocess_transactions, prepare_for_forecasting
from src.forecasting_model import ForecastingModel
from src.cash_advance_logic import CashAdvanceCalculator

# Define Pydantic models for API request/response
class MerchantId(BaseModel):
    anonymous_uu_id: str

class ForecastResponse(BaseModel):
    anonymous_uu_id: str
    forecasted_sales: List[float]

class CashAdvanceOfferResponse(BaseModel):
    anonymous_uu_id: str
    is_eligible: bool
    advance_amount: float = 0.0 # Default to 0 if not eligible

# Initialize FastAPI app
app = FastAPI(
    title="Merchant Sales Forecaster API",
    description="API for merchant sales forecasting and cash advance eligibility/amount calculation.",
    version="1.0.0"
)

# Global variables to store SparkSession, models, and data
# These will be initialized on application startup
spark = None
forecasting_model_instance = None
cash_advance_calculator_instance = None
historical_data_for_forecasting = None

# Module-level variable for data path (can be overridden for testing)
_data_path = None

def get_data_path():
    if _data_path is not None:
        return _data_path
    # Default data path for production
    return os.path.join(os.path.dirname(__file__), "..", "monthly_transactions.csv") 

def set_data_path(path: str):
    # Function to set data path for testing
    global _data_path
    _data_path = path

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "forecasting_pipeline_model")

@app.on_event("startup")
async def startup_event():
    global spark, forecasting_model_instance, cash_advance_calculator_instance, historical_data_for_forecasting
    
    # 1. Initialize Spark Session
    spark = get_or_create_spark_session()
    app.state.spark = spark # Store in app state for access in routes if needed

    # 2. Load and Preprocess Data (historical data still needed for feature generation and eligibility)
    DATA_PATH = get_data_path()
    
    print(f"Loading historical data from {DATA_PATH}...")
    raw_df = load_data(spark, DATA_PATH)
    preprocessed_df = preprocess_transactions(raw_df)
    historical_data_for_forecasting = prepare_for_forecasting(spark, preprocessed_df)
    app.state.historical_data_for_forecasting = historical_data_for_forecasting.cache() # Cache for performance
    print("Historical data loaded and preprocessed.")

    # 3. Load Pre-trained Forecasting Model (instead of training at startup)
    print(f"Loading pre-trained forecasting model from {MODEL_PATH}...")
    forecasting_model_instance = ForecastingModel.load(spark, MODEL_PATH)
    app.state.forecasting_model = forecasting_model_instance
    print("Forecasting model loaded.")

    # 4. Initialize Cash Advance Calculator
    cash_advance_calculator_instance = CashAdvanceCalculator(spark)
    app.state.cash_advance_calculator = cash_advance_calculator_instance

    print("Application startup complete: SparkSession, models, and data initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    global spark
    if spark:
        spark.stop()
        print("SparkSession stopped.")

@app.post("/forecast", response_model=ForecastResponse)
async def get_sales_forecast(merchant: MerchantId):
    """
    Forecasts the monthly sales revenue for a given merchant for the next 6 months.
    """
    merchant_id = merchant.anonymous_uu_id

    # Check if merchant exists in historical data
    if app.state.historical_data_for_forecasting.filter(col("anonymous_uu_id") == merchant_id).count() == 0:
        raise HTTPException(status_code=404, detail=f"Merchant ID {merchant_id} not found in historical data.")

    # Call the forecasting model to predict sales
    forecast_df = app.state.forecasting_model.predict(
        spark=app.state.spark,
        historical_df=app.state.historical_data_for_forecasting.filter(col("anonymous_uu_id") == merchant_id),
        months_to_forecast=6
    )
    forecasted_sales = [row.prediction for row in forecast_df.collect()]

    return ForecastResponse(anonymous_uu_id=merchant_id, forecasted_sales=forecasted_sales)

@app.post("/advance_offer", response_model=CashAdvanceOfferResponse)
async def get_cash_advance_offer(merchant: MerchantId):
    """
    Determines if a merchant is eligible for a cash advance and calculates the offer amount.
    """
    merchant_id = merchant.anonymous_uu_id

    # Check if merchant exists in historical data
    if app.state.historical_data_for_forecasting.filter(col("anonymous_uu_id") == merchant_id).count() == 0:
        raise HTTPException(status_code=404, detail=f"Merchant ID {merchant_id} not found in historical data.")

    # Get the latest month from the historical data for eligibility calculation context
    latest_historical_month = app.state.historical_data_for_forecasting.agg(max("transaction_month")).head()[0]

    # 1. Calculate historical features for eligibility
    historical_features = app.state.cash_advance_calculator._calculate_historical_features(
        df=app.state.historical_data_for_forecasting,
        as_of_date=latest_historical_month
    ).filter(col("anonymous_uu_id") == merchant_id)

    if historical_features.count() == 0:
        # This could happen if a merchant is in raw data but has no valid history for features
        return CashAdvanceOfferResponse(anonymous_uu_id=merchant_id, is_eligible=False, advance_amount=0.0)

    # 2. Determine eligibility
    eligibility_df = app.state.cash_advance_calculator.determine_eligibility(historical_features)
    is_eligible = eligibility_df.select("is_eligible").head()[0] if eligibility_df.count() > 0 else False

    advance_amount = 0.0
    if is_eligible:
        # 3. If eligible, forecast sales for the next 6 months
        forecast_df = app.state.forecasting_model.predict(
            spark=app.state.spark,
            historical_df=app.state.historical_data_for_forecasting.filter(col("anonymous_uu_id") == merchant_id),
            months_to_forecast=6
        )
        forecasted_sales = [row.prediction for row in forecast_df.collect()]
        # 4. Calculate cash advance amount
        advance_amount = app.state.cash_advance_calculator.calculate_advance_amount(
            merchant_id=merchant_id, # Not strictly needed for amount, but good for signature
            forecasted_sales=forecasted_sales
        )

    return CashAdvanceOfferResponse(anonymous_uu_id=merchant_id, is_eligible=is_eligible, advance_amount=advance_amount)
