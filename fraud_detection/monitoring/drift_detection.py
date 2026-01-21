"""
Model monitoring and drift detection.

Tracks:
- Data drift (feature distribution changes)
- Concept drift (model performance degradation)
- Prediction drift (output distribution changes)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path

from fraud_detection.utils import Config, setup_logger

logger = setup_logger(__name__)


class DriftDetector:
    """
    Detect data and concept drift in production.
    
    Uses statistical tests to identify when model retraining is needed.
    """
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize drift detector with reference data.
        
        Args:
            reference_data: Training data statistics as baseline
        """
        self.reference_stats = self._compute_statistics(reference_data)
        self.drift_threshold = Config.DRIFT_THRESHOLD
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute statistical summary of data"""
        stats = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q50': df[col].quantile(0.50),
                'q75': df[col].quantile(0.75)
            }
        
        return stats
    
    def detect_feature_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect drift in feature distributions.
        
        Uses Population Stability Index (PSI) for each feature.
        
        Args:
            current_data: Recent production data
            
        Returns:
            Dictionary with drift scores per feature
        """
        drift_scores = {}
        
        current_stats = self._compute_statistics(current_data)
        
        for feature, ref_stats in self.reference_stats.items():
            if feature not in current_stats:
                continue
            
            curr_stats = current_stats[feature]
            
            # Simple drift metric: normalized difference in means
            mean_diff = abs(curr_stats['mean'] - ref_stats['mean'])
            normalized_diff = mean_diff / (ref_stats['std'] + 1e-5)
            
            drift_scores[feature] = {
                'drift_score': normalized_diff,
                'is_drifting': normalized_diff > self.drift_threshold,
                'reference_mean': ref_stats['mean'],
                'current_mean': curr_stats['mean']
            }
        
        return drift_scores
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Dict:
        """
        Detect drift in prediction distributions.
        
        Args:
            reference_predictions: Historical predictions
            current_predictions: Recent predictions
            
        Returns:
            Drift analysis
        """
        from scipy.stats import ks_2samp
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(reference_predictions, current_predictions)
        
        return {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'is_drifting': p_value < 0.05,  # Significant difference
            'reference_mean': reference_predictions.mean(),
            'current_mean': current_predictions.mean()
        }


class PerformanceMonitor:
    """Monitor model performance in production"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """Compute performance metrics"""
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            average_precision_score,
            confusion_matrix
        )
        
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'pr_auc': average_precision_score(y_true, y_proba),
            'fraud_rate': y_true.mean(),
            'prediction_rate': y_pred.mean(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        })
        
        return metrics
    
    def log_metrics(self, metrics: Dict):
        """Log metrics to file"""
        import json
        
        metrics_file = self.log_dir / "performance_metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        logger.info(f"Metrics logged: PR-AUC={metrics['pr_auc']:.4f}")
    
    def check_performance_degradation(
        self,
        current_metrics: Dict,
        baseline_metrics: Dict,
        threshold: float = 0.05
    ) -> bool:
        """
        Check if performance has degraded significantly.
        
        Args:
            current_metrics: Recent performance metrics
            baseline_metrics: Training/baseline metrics
            threshold: Acceptable degradation threshold
            
        Returns:
            True if performance has degraded
        """
        pr_auc_drop = baseline_metrics['pr_auc'] - current_metrics['pr_auc']
        
        if pr_auc_drop > threshold:
            logger.warning(
                f"Performance degradation detected! "
                f"PR-AUC dropped by {pr_auc_drop:.4f}"
            )
            return True
        
        return False


if __name__ == "__main__":
    # Example monitoring workflow
    logger.info("Monitoring example")
    
    # Load reference data
    data_path = Config.get_data_path("transactions.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        
        # Split into reference and current
        split_idx = int(len(df) * 0.8)
        reference_df = df.iloc[:split_idx]
        current_df = df.iloc[split_idx:]
        
        # Initialize drift detector
        detector = DriftDetector(reference_df[['amount', 'distance_from_home']])
        
        # Detect drift
        drift_results = detector.detect_feature_drift(current_df[['amount', 'distance_from_home']])
        
        print("\n" + "="*60)
        print("DRIFT DETECTION RESULTS")
        print("="*60)
        for feature, result in drift_results.items():
            status = "⚠️  DRIFTING" if result['is_drifting'] else "✓ Stable"
            print(f"{feature}: {status}")
            print(f"  Drift score: {result['drift_score']:.4f}")
            print(f"  Reference mean: {result['reference_mean']:.2f}")
            print(f"  Current mean: {result['current_mean']:.2f}")
        print("="*60)
