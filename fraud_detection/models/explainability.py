"""
Model explainability using SHAP.

Provides interpretable explanations for fraud predictions,
critical for regulatory compliance and business trust.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from fraud_detection.utils import Config, setup_logger

logger = setup_logger(__name__)


class FraudExplainer:
    """
    SHAP-based explainability for fraud detection models.
    
    Provides:
    1. Global feature importance
    2. Local explanations for individual predictions
    3. Reason codes for high-risk transactions
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def fit(self, X_background: pd.DataFrame, sample_size: int = 100):
        """
        Fit SHAP explainer on background data.
        
        Args:
            X_background: Background dataset for SHAP
            sample_size: Number of samples to use (for efficiency)
        """
        logger.info("Initializing SHAP explainer...")
        
        # Sample background data for efficiency
        if len(X_background) > sample_size:
            background_sample = X_background.sample(n=sample_size, random_state=42)
        else:
            background_sample = X_background
        
        # Create TreeExplainer (efficient for tree-based models)
        self.explainer = shap.TreeExplainer(self.model)
        
        logger.info("SHAP explainer ready!")
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        top_n: int = 10
    ) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            X: Single transaction features (1 row DataFrame)
            top_n: Number of top contributing features to return
            
        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Get base value (expected prediction)
        base_value = self.explainer.expected_value
        
        # Get prediction
        import xgboost as xgb
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        prediction = self.model.predict(dtest)[0]
        
        # Create explanation DataFrame
        explanation_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': X.iloc[0].values,
            'shap_value': shap_values[0]
        })
        
        # Sort by absolute SHAP value
        explanation_df['abs_shap'] = explanation_df['shap_value'].abs()
        explanation_df = explanation_df.sort_values('abs_shap', ascending=False)
        
        # Top contributing features
        top_features = explanation_df.head(top_n)
        
        # Generate reason codes
        reason_codes = self._generate_reason_codes(top_features)
        
        return {
            'prediction': prediction,
            'base_value': base_value,
            'top_features': top_features[['feature', 'value', 'shap_value']].to_dict('records'),
            'reason_codes': reason_codes
        }
    
    def _generate_reason_codes(self, top_features: pd.DataFrame) -> List[str]:
        """
        Generate human-readable reason codes.
        
        Args:
            top_features: DataFrame of top contributing features
            
        Returns:
            List of reason code strings
        """
        reason_codes = []
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            value = row['value']
            shap_val = row['shap_value']
            
            # Only include features that increase fraud risk
            if shap_val > 0:
                if 'amount' in feature.lower():
                    if 'deviation' in feature.lower():
                        reason_codes.append(f"Unusual transaction amount (deviation: {value:.2f})")
                    elif value > 500:
                        reason_codes.append(f"High transaction amount (${value:.2f})")
                
                elif 'time_since_last' in feature.lower():
                    if value < 5:
                        reason_codes.append(f"Rapid transaction (only {value:.1f} minutes since last)")
                
                elif 'distance' in feature.lower():
                    if value > 100:
                        reason_codes.append(f"Transaction far from home ({value:.0f} km)")
                
                elif 'late_night' in feature.lower() and value == 1:
                    reason_codes.append("Transaction at unusual hour (late night)")
                
                elif 'terminal_fraud_rate' in feature.lower():
                    if value > 0.01:
                        reason_codes.append(f"High-risk merchant (fraud rate: {value:.2%})")
                
                elif 'tx_count' in feature.lower():
                    if value > 10:
                        reason_codes.append(f"High transaction frequency ({int(value)} recent transactions)")
        
        return reason_codes[:5]  # Return top 5 reasons
    
    def plot_feature_importance(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot global feature importance.
        
        Args:
            X: Dataset to compute SHAP values on
            max_display: Maximum features to display
            save_path: Path to save plot (optional)
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        logger.info("Computing SHAP values for feature importance...")
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def plot_waterfall(
        self,
        X: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot waterfall chart for a single prediction.
        
        Args:
            X: Single transaction (1 row DataFrame)
            save_path: Path to save plot (optional)
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        shap_values = self.explainer.shap_values(X)
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=self.explainer.expected_value,
            data=X.iloc[0].values,
            feature_names=self.feature_names
        )
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Waterfall plot saved to {save_path}")
        
        plt.close()


if __name__ == "__main__":
    # Example usage
    import joblib
    from fraud_detection.models.train import FraudDetectionModel
    
    # Load model
    model_path = Config.get_model_path("fraud_detector.joblib")
    if model_path.exists():
        model_artifact = joblib.load(model_path)
        model = model_artifact['model']
        feature_names = model_artifact['feature_names']
        
        # Load test data
        data_path = Config.get_data_path("transactions.csv")
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        
        # Get sample
        from fraud_detection.features import FraudFeatureEngineer
        feature_engineer = FraudFeatureEngineer()
        features = feature_engineer.create_features(df.head(1000), is_training=False)
        X, y = feature_engineer.prepare_for_modeling(features)
        
        # Initialize explainer
        explainer = FraudExplainer(model, feature_names)
        explainer.fit(X, sample_size=100)
        
        # Explain a fraud case
        fraud_idx = y[y == 1].index[0]
        explanation = explainer.explain_prediction(X.iloc[[fraud_idx]])
        
        print("\n" + "="*60)
        print("FRAUD PREDICTION EXPLANATION")
        print("="*60)
        print(f"Fraud Probability: {explanation['prediction']:.4f}")
        print(f"\nReason Codes:")
        for i, reason in enumerate(explanation['reason_codes'], 1):
            print(f"  {i}. {reason}")
        print("="*60)
    else:
        logger.error("No trained model found. Run train.py first.")
