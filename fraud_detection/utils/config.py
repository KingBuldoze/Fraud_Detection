"""
Configuration management for fraud detection system
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """Central configuration management"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Data generation
    NUM_CUSTOMERS = 10000
    NUM_TERMINALS = 1000
    SIMULATION_DAYS = 90
    FRAUD_RATIO = 0.005  # 0.5% fraud rate (realistic)
    
    # Feature engineering
    AGGREGATION_WINDOWS = {
        "1h": 1,
        "24h": 24,
        "7d": 168
    }
    
    # Model training
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42
    
    # XGBoost parameters
    XGBOOST_PARAMS = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Imbalance handling
    SCALE_POS_WEIGHT = 100  # Calculated as neg/pos ratio
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    MAX_LATENCY_MS = 200
    
    # Monitoring
    DRIFT_THRESHOLD = 0.1
    PERFORMANCE_ALERT_THRESHOLD = 0.05
    
    # Explainability
    SHAP_SAMPLE_SIZE = 100
    TOP_FEATURES_TO_EXPLAIN = 10
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_custom_config(cls, config_path: str) -> Dict[str, Any]:
        """Load custom configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def get_model_path(cls, model_name: str = "fraud_detector.joblib") -> Path:
        """Get path for saving/loading models"""
        return cls.MODELS_DIR / model_name
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get path for data files"""
        return cls.DATA_DIR / filename
