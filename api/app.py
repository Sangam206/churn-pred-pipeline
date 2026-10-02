"""
FastAPI application for Customer Churn Prediction.

Serves the trained XGBoost model with full preprocessing pipeline.
Accepts raw customer features and returns churn predictions.
"""

import os
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("churn-api")

# ---------------------------------------------------------------------------
# Paths (resolve relative to project root, not api/)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "preprocessor.pkl")

# ---------------------------------------------------------------------------
# Global model holders
# ---------------------------------------------------------------------------
model = None
preprocessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor on startup."""
    global model, preprocessor

    logger.info("Loading model from %s", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully (%s)", type(model).__name__)

    logger.info("Loading preprocessor from %s", PREPROCESSOR_PATH)
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("Preprocessor loaded successfully")

    yield  # app is running

    logger.info("Shutting down – releasing resources")
    model = None
    preprocessor = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predict whether an e-commerce customer will churn based on their "
        "behavioral and demographic features. Powered by an XGBoost model "
        "with Optuna-tuned hyperparameters."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """Raw customer features – exactly as they appear in the original dataset
    (minus City, which is dropped during preprocessing, and Churned which is
    the target)."""

    Age: float = Field(..., ge=0, le=120, description="Customer age")
    Gender: str = Field(..., description="Gender: Male, Female, or Other")
    Country: str = Field(
        ...,
        description="Country: Australia, Canada, France, Germany, India, Japan, UK, USA",
    )
    Membership_Years: float = Field(..., ge=0, description="Years of membership")
    Login_Frequency: float = Field(..., ge=0, description="Login frequency")
    Session_Duration_Avg: float = Field(..., ge=0, description="Average session duration (minutes)")
    Pages_Per_Session: float = Field(..., ge=0, description="Average pages viewed per session")
    Cart_Abandonment_Rate: float = Field(..., ge=0, le=100, description="Cart abandonment rate (%)")
    Wishlist_Items: float = Field(..., ge=0, description="Number of wishlist items")
    Total_Purchases: float = Field(..., ge=0, description="Total number of purchases")
    Average_Order_Value: float = Field(..., ge=0, description="Average order value ($)")
    Days_Since_Last_Purchase: float = Field(..., ge=0, description="Days since last purchase")
    Discount_Usage_Rate: float = Field(..., ge=0, le=100, description="Discount usage rate (%)")
    Returns_Rate: float = Field(..., ge=0, description="Product returns rate")
    Email_Open_Rate: float = Field(..., ge=0, description="Email open rate (%)")
    Customer_Service_Calls: float = Field(..., ge=0, description="Number of customer service calls")
    Product_Reviews_Written: float = Field(..., ge=0, description="Number of product reviews written")
    Social_Media_Engagement_Score: float = Field(..., ge=0, description="Social media engagement score")
    Mobile_App_Usage: float = Field(..., ge=0, description="Mobile app usage score")
    Payment_Method_Diversity: float = Field(..., ge=0, description="Number of distinct payment methods used")
    Lifetime_Value: float = Field(..., ge=0, description="Customer lifetime value ($)")
    Credit_Balance: float = Field(..., ge=0, description="Credit balance ($)")
    Signup_Quarter: str = Field(..., description="Signup quarter: Q1, Q2, Q3, or Q4")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Single prediction result."""
    churned: bool = Field(..., description="True if the customer is predicted to churn")
    churn_probability: float = Field(
        ..., ge=0, le=1, description="Probability of churning (0-1)"
    )
    confidence: str = Field(..., description="Confidence level: Low, Medium, or High")


class BatchPredictionRequest(BaseModel):
    """Batch prediction input."""
    customers: list[CustomerFeatures] = Field(
        ..., min_length=1, max_length=1000, description="List of customers (max 1000)"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction result."""
    predictions: list[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    model_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _preprocess_and_predict(customers: list[CustomerFeatures]):
    """Convert raw features → DataFrame → preprocessed array → predictions."""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503, detail="Model or preprocessor not loaded yet."
        )

    # Build DataFrame from raw input (same column order used during training)
    raw_data = [c.model_dump() for c in customers]
    df = pd.DataFrame(raw_data)

    # Apply the saved ColumnTransformer (OneHotEncoder + StandardScaler)
    try:
        X = preprocessor.transform(df)
    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        raise HTTPException(
            status_code=422,
            detail=f"Preprocessing failed – check your input values. Error: {e}",
        )

    # Make predictions
    probabilities = model.predict_proba(X)[:, 1]
    predictions_binary = (probabilities >= 0.5).astype(int)

    results = []
    for pred, prob in zip(predictions_binary, probabilities):
        prob_float = float(prob)
        if prob_float >= 0.8 or prob_float <= 0.2:
            confidence = "High"
        elif prob_float >= 0.6 or prob_float <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        results.append(
            PredictionResponse(
                churned=bool(pred),
                churn_probability=round(prob_float, 4),
                confidence=confidence,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

from fastapi.responses import FileResponse

@app.get("/", tags=["General"])
async def root():
    """API welcome message."""
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "version": "1.0.0",
        "ui": "/ui",
    }


@app.get("/ui", response_class=FileResponse, tags=["General"])
async def serve_ui():
    """Serve the web frontend dashboard."""
    frontend_path = os.path.join(BASE_DIR, "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="Frontend HTML file not found.")


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check whether the model and preprocessor are loaded."""
    return HealthResponse(
        status="healthy" if model and preprocessor else "degraded",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        model_type=type(model).__name__ if model else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(customer: CustomerFeatures):
    """Predict churn for a single customer."""
    logger.info("Single prediction request received")
    results = _preprocess_and_predict([customer])
    return results[0]


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for a batch of customers (max 1000)."""
    logger.info("Batch prediction request received (%d customers)", len(request.customers))
    results = _preprocess_and_predict(request.customers)
    return BatchPredictionResponse(predictions=results, total=len(results))


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return metadata about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    feature_names = model.get_booster().feature_names

    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "objective": model.objective,
    }
