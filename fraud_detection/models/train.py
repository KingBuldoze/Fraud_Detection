"""
Model training pipeline for fraud detection.

Implements:
- XGBoost with class imbalance handling
- Proper evaluation metrics for fraud (PR-AUC, not ROC-AUC)
- Model persistence and versioning
- Cost-based threshold optimization
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

from fraud_detection.utils import Config, setup_logger
from fraud_detection.data import TransactionGenerator
from fraud_detection.features import FraudFeatureEngineer, create_train_test_split

logger = setup_logger(__name__)


class FraudDetectionModel:
    """
    Production fraud detection model with XGBoost.
    
    Key design decisions:
    1. Use PR-AUC instead of ROC-AUC (handles imbalance)
    2. Optimize threshold based on business costs
    3. Track feature importance for interpretability
    """
    
    def __init__(self, model_params: Optional[Dict] = None):
        """
        Initialize fraud detection model.
        
        Args:
            model_params: XGBoost parameters (uses Config defaults if None)
        """
        self.model_params = model_params or Config.XGBOOST_PARAMS.copy()
        
        # Calculate scale_pos_weight from data if not provided
        if 'scale_pos_weight' not in self.model_params:
            self.model_params['scale_pos_weight'] = Config.SCALE_POS_WEIGHT
        
        self.model = None
        self.feature_names = None
        self.threshold = 0.5  # Default, will be optimized
        self.training_metrics = {}
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 20
    ) -> Dict:
        """
        Train XGBoost model with validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting model training...")
        logger.info(f"Training samples: {len(X_train)}, Fraud rate: {y_train.mean():.4%}")
        
        self.feature_names = X_train.columns.tolist()
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'val'))
            logger.info(f"Validation samples: {len(X_val)}, Fraud rate: {y_val.mean():.4%}")
        
        # Train model
        self.model = xgb.train(
            self.model_params,
            dtrain,
            num_boost_round=self.model_params.get('n_estimators', 200),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            verbose_eval=20
        )
        
        logger.info("Training complete!")
        
        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train, dataset_name='train')
        self.training_metrics['train'] = train_metrics
        
        # Evaluate on validation set
        if X_val is not None:
            val_metrics = self.evaluate(X_val, y_val, dataset_name='validation')
            self.training_metrics['validation'] = val_metrics
        
        return self.training_metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict fraud labels using threshold.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold (uses optimized if None)
            
        Returns:
            Binary predictions (0=legitimate, 1=fraud)
        """
        proba = self.predict_proba(X)
        threshold = threshold or self.threshold
        return (proba >= threshold).astype(int)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = 'test'
    ) -> Dict:
        """
        Comprehensive evaluation with fraud-specific metrics.
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name for logging
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on {dataset_name} set...")
        
        # Get predictions
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        # Precision-Recall metrics (critical for imbalanced data)
        pr_auc = average_precision_score(y, y_proba)
        
        # ROC-AUC (for comparison, but less meaningful for imbalanced data)
        roc_auc = roc_auc_score(y, y_proba)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # False positive rate (important for fraud - don't annoy customers)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'threshold': self.threshold
        }
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"{dataset_name.upper()} SET EVALUATION")
        logger.info(f"{'='*60}")
        logger.info(f"PR-AUC (Primary Metric): {pr_auc:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"False Positive Rate: {fpr:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn:,} | FP: {fp:,}")
        logger.info(f"  FN: {fn:,} | TP: {tp:,}")
        logger.info(f"{'='*60}\n")
        
        return metrics
    
    def optimize_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cost_fp: float = 1.0,
        cost_fn: float = 100.0,
        target_precision: Optional[float] = None
    ) -> float:
        """
        Optimize classification threshold based on business costs.
        
        Args:
            X: Features
            y: True labels
            cost_fp: Cost of false positive (investigating legitimate transaction)
            cost_fn: Cost of false negative (missing fraud)
            target_precision: If set, find threshold for this precision level
            
        Returns:
            Optimal threshold
        """
        logger.info("Optimizing classification threshold...")
        
        y_proba = self.predict_proba(X)
        
        if target_precision is not None:
            # Find threshold that achieves target precision
            precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
            
            # Find threshold where precision >= target
            valid_idx = np.where(precisions >= target_precision)[0]
            if len(valid_idx) > 0:
                # Among valid thresholds, choose one with highest recall
                best_idx = valid_idx[np.argmax(recalls[valid_idx])]
                optimal_threshold = thresholds[best_idx]
                logger.info(f"Threshold for {target_precision:.2%} precision: {optimal_threshold:.4f}")
                logger.info(f"Achieved recall: {recalls[best_idx]:.4f}")
            else:
                logger.warning(f"Could not achieve {target_precision:.2%} precision")
                optimal_threshold = 0.5
        else:
            # Optimize based on cost
            precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
            
            # Calculate expected cost for each threshold
            costs = []
            for i, threshold in enumerate(thresholds):
                y_pred = (y_proba >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
                
                total_cost = (fp * cost_fp) + (fn * cost_fn)
                costs.append(total_cost)
            
            # Find threshold with minimum cost
            optimal_idx = np.argmin(costs)
            optimal_threshold = thresholds[optimal_idx]
            
            logger.info(f"Optimal threshold (cost-based): {optimal_threshold:.4f}")
            logger.info(f"Precision: {precisions[optimal_idx]:.4f}")
            logger.info(f"Recall: {recalls[optimal_idx]:.4f}")
        
        self.threshold = optimal_threshold
        return optimal_threshold
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance_dict = self.model.get_score(importance_type='gain')
        
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, path: Optional[Path] = None):
        """Save model to disk"""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Config.get_model_path(f"fraud_model_{timestamp}.joblib")
        
        model_artifact = {
            'model': self.model,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'model_params': self.model_params,
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_artifact, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'FraudDetectionModel':
        """Load model from disk"""
        model_artifact = joblib.load(path)
        
        instance = cls(model_params=model_artifact['model_params'])
        instance.model = model_artifact['model']
        instance.feature_names = model_artifact['feature_names']
        instance.threshold = model_artifact['threshold']
        instance.training_metrics = model_artifact.get('training_metrics', {})
        
        logger.info(f"Model loaded from {path}")
        return instance


def train_fraud_model():
    """Main training pipeline"""
    
    Config.ensure_directories()
    
    # Load data
    data_path = Config.get_data_path("transactions.csv")
    
    if not data_path.exists():
        logger.info("Generating transaction data...")
        generator = TransactionGenerator(
            num_customers=Config.NUM_CUSTOMERS,
            num_terminals=Config.NUM_TERMINALS,
            simulation_days=Config.SIMULATION_DAYS,
            fraud_ratio=Config.FRAUD_RATIO
        )
        df = generator.generate_dataset()
        df.to_csv(data_path, index=False)
    else:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
    
    # Train/test split (temporal)
    train_df, test_df = create_train_test_split(df, test_size=Config.TEST_SIZE, time_based=True)
    
    # Further split train into train/validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=Config.VALIDATION_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=train_df['is_fraud']
    )
    
    # Feature engineering
    feature_engineer = FraudFeatureEngineer()
    
    logger.info("Engineering features for training set...")
    train_features = feature_engineer.create_features(train_df, is_training=True)
    X_train, y_train = feature_engineer.prepare_for_modeling(train_features)
    
    logger.info("Engineering features for validation set...")
    val_features = feature_engineer.create_features(val_df, is_training=True)
    X_val, y_val = feature_engineer.prepare_for_modeling(val_features)
    
    logger.info("Engineering features for test set...")
    test_features = feature_engineer.create_features(test_df, is_training=False)
    X_test, y_test = feature_engineer.prepare_for_modeling(test_features)
    
    # Train model
    model = FraudDetectionModel()
    model.train(X_train, y_train, X_val, y_val)
    
    # Optimize threshold (target 95% precision)
    model.optimize_threshold(X_val, y_val, target_precision=0.95)
    
    # Final evaluation on test set
    test_metrics = model.evaluate(X_test, y_test, dataset_name='test')
    
    # Feature importance
    importance_df = model.get_feature_importance(top_n=20)
    logger.info("\nTop 20 Most Important Features:")
    logger.info(importance_df.to_string(index=False))
    
    # Save model
    model.save()
    
    # Save feature engineer
    feature_engineer_path = Config.MODELS_DIR / "feature_engineer.joblib"
    joblib.dump(feature_engineer, feature_engineer_path)
    logger.info(f"Feature engineer saved to {feature_engineer_path}")
    
    return model, test_metrics


if __name__ == "__main__":
    train_fraud_model()
