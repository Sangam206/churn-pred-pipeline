"""Unit and integration tests for the Churn Prediction API and Frontend serving."""
import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    """TestClient fixture with lifespan context manager active."""
    with TestClient(app) as c:
        yield c


def test_root_ui_endpoint(client):
    """Verify root endpoint returns API status message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Customer Churn Prediction API" in response.text


def test_ui_endpoint(client):
    """Verify /ui endpoint serves HTML UI."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "Customer Churn" in response.text


def test_health_check(client):
    """Verify /health endpoint returns healthy status and loaded model info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_single_prediction(client):
    """Verify /predict endpoint returns prediction output."""
    payload = {
        "Age": 43.0,
        "Gender": "Male",
        "Country": "France",
        "Membership_Years": 2.9,
        "Login_Frequency": 14.0,
        "Session_Duration_Avg": 27.4,
        "Pages_Per_Session": 6.0,
        "Cart_Abandonment_Rate": 50.6,
        "Wishlist_Items": 3.0,
        "Total_Purchases": 9.0,
        "Average_Order_Value": 94.72,
        "Days_Since_Last_Purchase": 34.0,
        "Discount_Usage_Rate": 46.40,
        "Returns_Rate": 2.0,
        "Email_Open_Rate": 17.9,
        "Customer_Service_Calls": 9.0,
        "Product_Reviews_Written": 4.0,
        "Social_Media_Engagement_Score": 16.3,
        "Mobile_App_Usage": 20.8,
        "Payment_Method_Diversity": 1.0,
        "Lifetime_Value": 953.33,
        "Credit_Balance": 2278.0,
        "Signup_Quarter": "Q1",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churned" in data
    assert "churn_probability" in data
    assert "confidence" in data


def test_batch_prediction(client):
    """Verify /predict/batch endpoint processes multiple customers."""
    payload = {
        "Age": 43.0,
        "Gender": "Male",
        "Country": "France",
        "Membership_Years": 2.9,
        "Login_Frequency": 14.0,
        "Session_Duration_Avg": 27.4,
        "Pages_Per_Session": 6.0,
        "Cart_Abandonment_Rate": 50.6,
        "Wishlist_Items": 3.0,
        "Total_Purchases": 9.0,
        "Average_Order_Value": 94.72,
        "Days_Since_Last_Purchase": 34.0,
        "Discount_Usage_Rate": 46.40,
        "Returns_Rate": 2.0,
        "Email_Open_Rate": 17.9,
        "Customer_Service_Calls": 9.0,
        "Product_Reviews_Written": 4.0,
        "Social_Media_Engagement_Score": 16.3,
        "Mobile_App_Usage": 20.8,
        "Payment_Method_Diversity": 1.0,
        "Lifetime_Value": 953.33,
        "Credit_Balance": 2278.0,
        "Signup_Quarter": "Q1",
    }
    batch = {
        "customers": [
            payload,
            {**payload, "Age": 22.0, "Country": "India", "Login_Frequency": 2.0},
        ]
    }
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2


def test_model_info(client):
    """Verify /model/info endpoint returns booster metadata."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "n_features" in data
