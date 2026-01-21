"""
Integration tests for fraud detection API
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from fraud_detection.api.app import app


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def sample_transaction():
    """Sample transaction for testing"""
    return {
        "transaction_id": "TX_TEST_001",
        "customer_id": 123,
        "terminal_id": 45,
        "amount": 99.99,
        "timestamp": datetime.utcnow().isoformat(),
        "merchant_category": "electronics",
        "distance_from_home": 10.5
    }


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_fraud_valid_request(client, sample_transaction):
    """Test fraud prediction with valid request"""
    response = client.post("/predict", json=sample_transaction)
    
    # May return 503 if model not loaded (acceptable in test environment)
    if response.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert "transaction_id" in data
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "reason_codes" in data
    assert "processing_time_ms" in data
    
    # Validate data types
    assert isinstance(data["is_fraud"], bool)
    assert isinstance(data["fraud_probability"], float)
    assert 0 <= data["fraud_probability"] <= 1
    assert data["risk_level"] in ["low", "medium", "high"]


def test_predict_fraud_invalid_amount(client, sample_transaction):
    """Test prediction with invalid amount"""
    sample_transaction["amount"] = -50  # Invalid negative amount
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 422  # Validation error


def test_predict_fraud_invalid_category(client, sample_transaction):
    """Test prediction with invalid merchant category"""
    sample_transaction["merchant_category"] = "invalid_category"
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 422


def test_model_info(client):
    """Test model info endpoint"""
    response = client.get("/model/info")
    
    if response.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "num_features" in data


def test_feature_importance(client):
    """Test feature importance endpoint"""
    response = client.get("/model/feature_importance")
    
    if response.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_batch_prediction(client, sample_transaction):
    """Test batch prediction endpoint"""
    transactions = [sample_transaction.copy() for i in range(3)]
    
    # Make unique IDs
    for i, tx in enumerate(transactions):
        tx["transaction_id"] = f"TX_TEST_{i:03d}"
    
    response = client.post("/predict/batch", json=transactions)
    
    if response.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_batch_prediction_size_limit(client, sample_transaction):
    """Test batch size limit"""
    transactions = [sample_transaction.copy() for _ in range(1001)]
    response = client.post("/predict/batch", json=transactions)
    assert response.status_code == 400  # Batch size exceeded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
