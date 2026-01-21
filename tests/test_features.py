"""
Unit tests for feature engineering
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from fraud_detection.features import FraudFeatureEngineer, create_train_test_split


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing"""
    np.random.seed(42)
    
    data = {
        'transaction_id': [f'TX{i:05d}' for i in range(100)],
        'customer_id': np.random.randint(0, 10, 100),
        'terminal_id': np.random.randint(0, 5, 100),
        'amount': np.random.uniform(10, 500, 100),
        'timestamp': [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(100)],
        'merchant_category': np.random.choice(['grocery', 'electronics', 'restaurant'], 100),
        'distance_from_home': np.random.uniform(0, 100, 100),
        'is_fraud': np.random.choice([0, 1], 100, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)


def test_feature_engineer_initialization():
    """Test feature engineer initialization"""
    fe = FraudFeatureEngineer()
    assert fe is not None
    assert hasattr(fe, 'feature_names')


def test_create_features(sample_data):
    """Test feature creation"""
    fe = FraudFeatureEngineer()
    features = fe.create_features(sample_data, is_training=True)
    
    # Check that features were added
    assert len(features.columns) > len(sample_data.columns)
    
    # Check specific features exist
    assert 'hour' in features.columns
    assert 'day_of_week' in features.columns
    assert 'is_weekend' in features.columns
    assert 'amount_log' in features.columns


def test_temporal_features(sample_data):
    """Test temporal feature generation"""
    fe = FraudFeatureEngineer()
    features = fe.create_features(sample_data, is_training=True)
    
    # Check temporal features
    assert 'hour' in features.columns
    assert features['hour'].min() >= 0
    assert features['hour'].max() <= 23
    
    assert 'is_weekend' in features.columns
    assert features['is_weekend'].isin([0, 1]).all()


def test_no_data_leakage(sample_data):
    """Test that training mode prevents data leakage"""
    fe = FraudFeatureEngineer()
    
    # In training mode, aggregations should only use past data
    features_train = fe.create_features(sample_data, is_training=True)
    
    # First transaction should have zero aggregations (no history)
    first_row = features_train.iloc[0]
    assert first_row['customer_tx_count_1h'] == 0


def test_train_test_split(sample_data):
    """Test train/test splitting"""
    train_df, test_df = create_train_test_split(sample_data, test_size=0.2, time_based=True)
    
    # Check sizes
    assert len(train_df) + len(test_df) == len(sample_data)
    assert len(test_df) == pytest.approx(len(sample_data) * 0.2, abs=1)
    
    # Check temporal ordering
    assert train_df['timestamp'].max() <= test_df['timestamp'].min()


def test_prepare_for_modeling(sample_data):
    """Test data preparation for modeling"""
    fe = FraudFeatureEngineer()
    features = fe.create_features(sample_data, is_training=True)
    
    X, y = fe.prepare_for_modeling(features)
    
    # Check shapes
    assert len(X) == len(sample_data)
    assert len(y) == len(sample_data)
    
    # Check no NaN values
    assert not X.isna().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
