import pytest
from fastapi.testclient import TestClient
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType
from datetime import datetime
import os

# Import the FastAPI app and data path setter from src.api
from src.api import app, set_data_path
from src.data_processing import preprocess_transactions, prepare_for_forecasting

# Path to a dummy CSV file for testing
dummy_data_path = "./tests/dummy_monthly_transactions.csv"

@pytest.fixture(scope="module")
def setup_test_data():
    """
    Creates a dummy CSV file for API testing and cleans it up afterward.
    """
    csv_content = """anonymous_uu_id,currency_code,country_code,transaction_month,sales_amount,transaction_count
merchant_A,USD,US,202201,8000.0,80
merchant_A,USD,US,202202,8500.0,85
merchant_A,USD,US,202203,9000.0,90
merchant_A,USD,US,202204,9500.0,95
merchant_A,USD,US,202205,10000.0,100
merchant_A,USD,US,202206,10500.0,105
merchant_A,USD,US,202207,11000.0,110
merchant_A,USD,US,202208,11500.0,115
merchant_A,USD,US,202209,12000.0,120
merchant_A,USD,US,202210,12500.0,125
merchant_A,USD,US,202211,13000.0,130
merchant_A,USD,US,202212,13500.0,135
merchant_A,USD,US,202301,14000.0,140
merchant_A,USD,US,202302,14500.0,145
merchant_A,USD,US,202303,15000.0,150
merchant_A,USD,US,202304,15500.0,155
merchant_A,USD,US,202305,16000.0,160
merchant_A,USD,US,202306,16500.0,165
merchant_B,USD,US,202305,1000.0,10
merchant_B,USD,US,202306,1050.0,11
merchant_C,USD,US,202201,100.0,5
merchant_C,USD,US,202202,100.0,5
merchant_C,USD,US,202203,100.0,5
merchant_C,USD,US,202204,100.0,5
merchant_C,USD,US,202205,100.0,5
merchant_C,USD,US,202206,100.0,5
merchant_C,USD,US,202207,100.0,5
merchant_C,USD,US,202208,100.0,5
merchant_C,USD,US,202209,100.0,5
merchant_C,USD,US,202210,100.0,5
merchant_C,USD,US,202211,100.0,5
merchant_C,USD,US,202212,100.0,5
merchant_D,USD,US,202201,10000.0,100
merchant_D,USD,US,202202,9500.0,95
merchant_D,USD,US,202203,9000.0,90
merchant_D,USD,US,202204,8500.0,85
merchant_D,USD,US,202205,8000.0,80
merchant_D,USD,US,202206,7500.0,75
merchant_D,USD,US,202207,5000.0,50
merchant_D,USD,US,202208,4000.0,40
merchant_D,USD,US,202209,3000.0,30
merchant_D,USD,US,202210,2000.0,20
merchant_D,USD,US,202211,1500.0,15
merchant_D,USD,US,202212,1000.0,10
"""
    with open(dummy_data_path, "w") as f:
        f.write(csv_content)
    yield
    os.remove(dummy_data_path)

class MockForecastingModel:
    """Mock forecasting model for testing that returns simple predictions without complex PySpark operations."""
    
    def _get_mock_predictions(self, merchant_id):
        """Helper method to get mock predictions based on merchant ID."""
        if merchant_id == "merchant_A":
            return [17000.0, 17500.0, 18000.0, 18500.0, 19000.0, 19500.0]
        elif merchant_id == "merchant_B":
            return [1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0]
        elif merchant_id == "merchant_C":
            return [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        elif merchant_id == "merchant_D":
            return [800.0, 700.0, 600.0, 500.0, 400.0, 300.0]
        else:
            return [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    
    def predict(self, spark, historical_df, months_to_forecast=6):
        # Simple mock that returns dummy predictions
        from pyspark.sql.types import StructType, StructField, DoubleType
        from pyspark.sql import Row
        
        # Return mock predictions based on merchant
        merchant_id = historical_df.select("anonymous_uu_id").first().anonymous_uu_id
        predictions = self._get_mock_predictions(merchant_id)
        
        # Create DataFrame with predictions
        schema = StructType([StructField("prediction", DoubleType(), True)])
        rows = [Row(prediction=pred) for pred in predictions]
        return spark.createDataFrame(rows, schema)
    
    def predict_fast(self, spark, historical_df, months_to_forecast=6):
        """Mock implementation of predict_fast method - same as predict for testing."""
        return self.predict(spark, historical_df, months_to_forecast)

@pytest.fixture(scope="module") 
def client_with_data(setup_test_data):
    """
    Provides a TestClient for the FastAPI app, ensuring data is loaded from the dummy file.
    Uses a mock forecasting model to avoid complex PySpark issues in testing.
    """
    # Set the data path for the API to use the dummy file
    set_data_path(dummy_data_path)
    
    # Override the forecasting model with a mock for testing
    original_model_class = None
    
    with TestClient(app) as client:
        # After startup, replace the model with our mock
        if hasattr(app.state, 'forecasting_model'):
            app.state.forecasting_model = MockForecastingModel()
        yield client


def test_forecast_endpoint_success(client_with_data):
    """
    Tests the /forecast endpoint for a successful prediction.
    """
    response = client_with_data.post(
        "/forecast",
        json={"anonymous_uu_id": "merchant_A"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["anonymous_uu_id"] == "merchant_A"
    assert len(data["forecasted_sales"]) == 6
    assert all(isinstance(s, float) for s in data["forecasted_sales"])

def test_forecast_endpoint_merchant_not_found(client_with_data):
    """
    Tests the /forecast endpoint for a merchant not found scenario.
    """
    response = client_with_data.post(
        "/forecast",
        json={"anonymous_uu_id": "non_existent_merchant"}
    )
    assert response.status_code == 404
    assert "Merchant ID non_existent_merchant not found" in response.json()["detail"]

def test_advance_offer_endpoint_eligible(client_with_data):
    """
    Tests the /advance_offer endpoint for an eligible merchant.
    """
    response = client_with_data.post(
        "/advance_offer",
        json={"anonymous_uu_id": "merchant_A"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["anonymous_uu_id"] == "merchant_A"
    assert data["is_eligible"] == True
    assert data["advance_amount"] > 0

def test_advance_offer_endpoint_not_eligible_insufficient_history(client_with_data):
    """
    Tests the /advance_offer endpoint for an ineligible merchant due to insufficient history.
    """
    response = client_with_data.post(
        "/advance_offer",
        json={"anonymous_uu_id": "merchant_B"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["anonymous_uu_id"] == "merchant_B"
    assert data["is_eligible"] == False
    assert data["advance_amount"] == 0.0

def test_advance_offer_endpoint_not_eligible_low_sales(client_with_data):
    """
    Tests the /advance_offer endpoint for an ineligible merchant due to low sales.
    """
    response = client_with_data.post(
        "/advance_offer",
        json={"anonymous_uu_id": "merchant_C"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["anonymous_uu_id"] == "merchant_C"
    assert data["is_eligible"] == False
    assert data["advance_amount"] == 0.0

def test_advance_offer_endpoint_not_eligible_declining_trend(client_with_data):
    """
    Tests the /advance_offer endpoint for an ineligible merchant due to declining sales trend.
    """
    response = client_with_data.post(
        "/advance_offer",
        json={"anonymous_uu_id": "merchant_D"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["anonymous_uu_id"] == "merchant_D"
    assert data["is_eligible"] == False
    assert data["advance_amount"] == 0.0

def test_advance_offer_endpoint_merchant_not_found(client_with_data):
    """
    Tests the /advance_offer endpoint for a merchant not found scenario.
    """
    response = client_with_data.post(
        "/advance_offer",
        json={"anonymous_uu_id": "non_existent_merchant"}
    )
    assert response.status_code == 404
    assert "Merchant ID non_existent_merchant not found" in response.json()["detail"]
