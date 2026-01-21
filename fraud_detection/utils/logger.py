"""
Logging utilities for fraud detection system
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class PerformanceLogger:
    """Log model performance metrics for monitoring"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = log_dir / "performance_metrics.jsonl"
    
    def log_prediction(
        self,
        transaction_id: str,
        prediction: float,
        features: dict,
        latency_ms: float
    ):
        """Log individual prediction for audit trail"""
        import json
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "transaction_id": transaction_id,
            "fraud_probability": prediction,
            "latency_ms": latency_ms,
            "features": features
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_batch_metrics(self, metrics: dict):
        """Log batch evaluation metrics"""
        import json
        
        metrics_with_timestamp = {
            "timestamp": datetime.utcnow().isoformat(),
            **metrics
        }
        
        batch_file = self.log_dir / "batch_metrics.jsonl"
        with open(batch_file, 'a') as f:
            f.write(json.dumps(metrics_with_timestamp) + '\n')
