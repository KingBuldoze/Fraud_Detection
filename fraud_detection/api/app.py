"""
Real-time fraud detection API using FastAPI.

Provides:
- Low-latency transaction scoring
- Explainable predictions
- Request validation
- Performance monitoring
"""

import time
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from fraud_detection.utils import Config, setup_logger, PerformanceLogger
from fraud_detection.models import FraudDetectionModel, FraudExplainer
from fraud_detection.features import FraudFeatureEngineer

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud detection with explainable AI",
    version="1.0.0"
)

# Global model and feature engineer (loaded on startup)
model: Optional[FraudDetectionModel] = None
feature_engineer: Optional[FraudFeatureEngineer] = None
explainer: Optional[FraudExplainer] = None
performance_logger: Optional[PerformanceLogger] = None


class TransactionRequest(BaseModel):
    """Request schema for fraud detection"""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: int = Field(..., description="Customer identifier")
    terminal_id: int = Field(..., description="Terminal/merchant identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    merchant_category: str = Field(..., description="Merchant category")
    distance_from_home: float = Field(..., ge=0, description="Distance from customer home (km)")
    
    @validator('merchant_category')
    def validate_category(cls, v):
        valid_categories = [
            'grocery', 'restaurant', 'gas_station', 'online_retail',
            'electronics', 'clothing', 'entertainment', 'travel', 'other'
        ]
        if v not in valid_categories:
            raise ValueError(f"Invalid category. Must be one of: {valid_categories}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TX00123456",
                "customer_id": 42,
                "terminal_id": 15,
                "amount": 125.50,
                "timestamp": "2024-01-15T14:30:00",
                "merchant_category": "electronics",
                "distance_from_home": 5.2
            }
        }


class FraudPrediction(BaseModel):
    """Response schema for fraud detection"""
    
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    reason_codes: List[str]
    processing_time_ms: float
    timestamp: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TX00123456",
                "is_fraud": False,
                "fraud_probability": 0.0234,
                "risk_level": "low",
                "reason_codes": [],
                "processing_time_ms": 45.2,
                "timestamp": "2024-01-15T14:30:05"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: datetime


@app.on_event("startup")
async def load_models():
    """Load models on application startup"""
    global model, feature_engineer, explainer, performance_logger
    
    logger.info("Loading fraud detection models...")
    
    Config.ensure_directories()
    
    # Load feature engineer
    feature_engineer_path = Config.MODELS_DIR / "feature_engineer.joblib"
    if feature_engineer_path.exists():
        feature_engineer = joblib.load(feature_engineer_path)
        logger.info("Feature engineer loaded")
    else:
        logger.warning("Feature engineer not found. Using new instance.")
        feature_engineer = FraudFeatureEngineer()
    
    # Load model (find latest)
    model_files = list(Config.MODELS_DIR.glob("fraud_model_*.joblib"))
    
    if model_files:
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        model = FraudDetectionModel.load(latest_model)
        logger.info(f"Model loaded from {latest_model}")
        
        # Initialize explainer
        # Note: In production, you'd load pre-computed SHAP values
        # For now, we'll initialize it lazily on first request
        explainer = None
    else:
        logger.error("No trained model found!")
        model = None
    
    # Initialize performance logger
    performance_logger = PerformanceLogger(Config.LOGS_DIR)
    
    logger.info("API ready!")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        timestamp=datetime.utcnow()
    )


@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict fraud for a transaction.
    
    Returns fraud probability, binary prediction, and explanation.
    """
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Engineer features
        # Note: In production, you'd fetch historical data from a feature store
        # For this demo, we'll use the transaction as-is with basic features
        features_df = feature_engineer.create_features(df, is_training=False)
        X, _ = feature_engineer.prepare_for_modeling(features_df)
        
        # Predict
        fraud_proba = model.predict_proba(X)[0]
        is_fraud = model.predict(X)[0]
        
        # Determine risk level
        if fraud_proba < 0.3:
            risk_level = "low"
        elif fraud_proba < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Get explanation (if available)
        reason_codes = []
        if explainer is not None and fraud_proba > 0.5:
            try:
                explanation = explainer.explain_prediction(X)
                reason_codes = explanation['reason_codes']
            except Exception as e:
                logger.warning(f"Explanation failed: {e}")
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log performance
        if performance_logger:
            performance_logger.log_prediction(
                transaction_id=transaction.transaction_id,
                prediction=float(fraud_proba),
                features=transaction_dict,
                latency_ms=processing_time_ms
            )
        
        # Check latency SLA
        if processing_time_ms > Config.MAX_LATENCY_MS:
            logger.warning(f"Latency SLA violated: {processing_time_ms:.2f}ms")
        
        return FraudPrediction(
            transaction_id=transaction.transaction_id,
            is_fraud=bool(is_fraud),
            fraud_probability=float(fraud_proba),
            risk_level=risk_level,
            reason_codes=reason_codes,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(transactions: List[TransactionRequest]):
    """
    Batch prediction endpoint for multiple transactions.
    
    More efficient for processing multiple transactions at once.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limited to 1000 transactions")
    
    results = []
    
    for transaction in transactions:
        try:
            result = await predict_fraud(transaction)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch prediction error for {transaction.transaction_id}: {e}")
            # Continue with other transactions
    
    return results


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost",
        "num_features": len(model.feature_names),
        "threshold": model.threshold,
        "training_metrics": model.training_metrics,
        "feature_names": model.feature_names[:20]  # Top 20 features
    }


@app.get("/model/feature_importance")
async def feature_importance():
    """Get feature importance scores"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    importance_df = model.get_feature_importance(top_n=20)
    return importance_df.to_dict('records')


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "fraud_detection.api.app:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,
        log_level="info"
    )
