"""
Feature engineering pipeline for fraud detection.

Implements production-grade feature engineering with:
- Temporal aggregations (rolling windows)
- Behavioral features (spending patterns)
- Risk indicators
- Proper train/test separation to prevent data leakage
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datetime import timedelta

from fraud_detection.utils import Config, setup_logger

logger = setup_logger(__name__)


class FraudFeatureEngineer:
    """
    Feature engineering for fraud detection with leak prevention.
    
    Key principles:
    1. All aggregations use only historical data (no future leakage)
    2. Features are computed the same way in training and inference
    3. Missing values handled explicitly
    """
    
    def __init__(self):
        self.feature_names = []
        self.aggregation_windows = Config.AGGREGATION_WINDOWS
    
    def create_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        Create all features for fraud detection.
        
        Args:
            df: Transaction dataframe with required columns
            is_training: If True, compute features for training (historical only)
                        If False, compute for inference (use all available data)
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic temporal features
        df = self._add_temporal_features(df)
        
        # Transaction-level features
        df = self._add_transaction_features(df)
        
        # Customer behavioral features (aggregations)
        df = self._add_customer_aggregations(df, is_training)
        
        # Terminal risk features
        df = self._add_terminal_features(df, is_training)
        
        # Velocity features (transaction frequency)
        df = self._add_velocity_features(df, is_training)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Business hours indicator
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Late night indicator (high fraud risk)
        df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
        
        self.feature_names.extend([
            'hour', 'day_of_week', 'day_of_month', 
            'is_weekend', 'is_business_hours', 'is_late_night'
        ])
        
        return df
    
    def _add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction-level features"""
        
        # Log transform of amount (handles skewness)
        df['amount_log'] = np.log1p(df['amount'])
        
        # Round amount indicator (fraudsters often use round numbers)
        df['is_round_amount'] = (df['amount'] % 10 == 0).astype(int)
        
        # Distance features
        df['distance_log'] = np.log1p(df['distance_from_home'])
        df['is_far_from_home'] = (df['distance_from_home'] > 100).astype(int)
        
        # Merchant category encoding (one-hot)
        category_dummies = pd.get_dummies(
            df['merchant_category'], 
            prefix='category'
        )
        df = pd.concat([df, category_dummies], axis=1)
        
        self.feature_names.extend([
            'amount_log', 'is_round_amount', 
            'distance_log', 'is_far_from_home'
        ])
        self.feature_names.extend(category_dummies.columns.tolist())
        
        return df
    
    def _add_customer_aggregations(
        self, 
        df: pd.DataFrame, 
        is_training: bool
    ) -> pd.DataFrame:
        """
        Add customer behavioral features using rolling aggregations.
        
        CRITICAL: Only use historical data to prevent leakage
        """
        
        df = df.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
        
        for window_name, window_hours in self.aggregation_windows.items():
            window_td = timedelta(hours=window_hours)
            
            # For each transaction, look back at previous transactions
            agg_features = []
            
            for idx, row in df.iterrows():
                customer_id = row['customer_id']
                current_time = row['timestamp']
                
                # Get historical transactions for this customer
                if is_training:
                    # Exclude current transaction (use only past)
                    mask = (
                        (df['customer_id'] == customer_id) &
                        (df['timestamp'] < current_time) &
                        (df['timestamp'] >= current_time - window_td)
                    )
                else:
                    # In inference, we can use all data up to current time
                    mask = (
                        (df['customer_id'] == customer_id) &
                        (df['timestamp'] <= current_time) &
                        (df['timestamp'] >= current_time - window_td)
                    )
                
                hist_txs = df[mask]
                
                # Compute aggregations
                agg_features.append({
                    f'customer_tx_count_{window_name}': len(hist_txs),
                    f'customer_amount_mean_{window_name}': hist_txs['amount'].mean() if len(hist_txs) > 0 else 0,
                    f'customer_amount_std_{window_name}': hist_txs['amount'].std() if len(hist_txs) > 0 else 0,
                    f'customer_amount_max_{window_name}': hist_txs['amount'].max() if len(hist_txs) > 0 else 0,
                })
            
            agg_df = pd.DataFrame(agg_features)
            df = pd.concat([df.reset_index(drop=True), agg_df], axis=1)
            
            self.feature_names.extend(agg_df.columns.tolist())
        
        # Deviation from customer's normal behavior
        if f'customer_amount_mean_24h' in df.columns:
            df['amount_deviation_24h'] = (
                df['amount'] - df['customer_amount_mean_24h']
            ) / (df['customer_amount_std_24h'] + 1e-5)
            self.feature_names.append('amount_deviation_24h')
        
        return df
    
    def _add_terminal_features(
        self, 
        df: pd.DataFrame, 
        is_training: bool
    ) -> pd.DataFrame:
        """Add terminal/merchant risk features"""
        
        # Terminal fraud rate (historical)
        terminal_fraud_rate = []
        
        for idx, row in df.iterrows():
            terminal_id = row['terminal_id']
            current_time = row['timestamp']
            
            if is_training:
                mask = (
                    (df['terminal_id'] == terminal_id) &
                    (df['timestamp'] < current_time)
                )
            else:
                mask = (
                    (df['terminal_id'] == terminal_id) &
                    (df['timestamp'] <= current_time)
                )
            
            hist_txs = df[mask]
            
            if len(hist_txs) > 0 and 'is_fraud' in hist_txs.columns:
                fraud_rate = hist_txs['is_fraud'].mean()
            else:
                fraud_rate = 0
            
            terminal_fraud_rate.append(fraud_rate)
        
        df['terminal_fraud_rate'] = terminal_fraud_rate
        self.feature_names.append('terminal_fraud_rate')
        
        return df
    
    def _add_velocity_features(
        self, 
        df: pd.DataFrame, 
        is_training: bool
    ) -> pd.DataFrame:
        """
        Add velocity features: time since last transaction.
        
        Rapid transactions are a strong fraud signal.
        """
        
        df = df.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
        
        # Time since last transaction for this customer
        df['time_since_last_tx'] = df.groupby('customer_id')['timestamp'].diff().dt.total_seconds() / 60
        
        # Fill first transaction for each customer
        df['time_since_last_tx'] = df['time_since_last_tx'].fillna(1440)  # 24 hours
        
        # Rapid transaction indicator
        df['is_rapid_tx'] = (df['time_since_last_tx'] < 5).astype(int)  # < 5 minutes
        
        self.feature_names.extend(['time_since_last_tx', 'is_rapid_tx'])
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        
        # Fill NaN with 0 for aggregation features (means no history)
        for col in df.columns:
            if col in self.feature_names and df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names"""
        return self.feature_names
    
    def prepare_for_modeling(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for modeling.
        
        Returns:
            X: Feature matrix
            y: Target variable
        """
        
        # Select only feature columns
        feature_cols = [col for col in self.feature_names if col in df.columns]
        
        X = df[feature_cols].copy()
        y = df['is_fraud'].copy() if 'is_fraud' in df.columns else None
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        
        return X, y


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_based: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: Full dataset
        test_size: Proportion for test set
        time_based: If True, use temporal split (more realistic)
                   If False, use random split
    
    Returns:
        train_df, test_df
    """
    
    if time_based:
        # Temporal split: train on earlier data, test on later
        df = df.sort_values('timestamp')
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Temporal split: train up to {train_df['timestamp'].max()}")
        logger.info(f"Test from {test_df['timestamp'].min()}")
    else:
        # Random split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=Config.RANDOM_STATE,
            stratify=df['is_fraud']
        )
    
    logger.info(f"Train set: {len(train_df)} samples, {train_df['is_fraud'].mean():.4%} fraud")
    logger.info(f"Test set: {len(test_df)} samples, {test_df['is_fraud'].mean():.4%} fraud")
    
    return train_df, test_df


if __name__ == "__main__":
    # Test feature engineering
    from fraud_detection.data import TransactionGenerator
    
    Config.ensure_directories()
    
    # Load or generate data
    data_path = Config.get_data_path("transactions.csv")
    
    if data_path.exists():
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
    else:
        logger.info("Generating new dataset...")
        generator = TransactionGenerator(
            num_customers=1000,
            num_terminals=100,
            simulation_days=30
        )
        df = generator.generate_dataset()
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df, test_size=0.2)
    
    # Engineer features
    feature_engineer = FraudFeatureEngineer()
    
    train_features = feature_engineer.create_features(train_df, is_training=True)
    test_features = feature_engineer.create_features(test_df, is_training=False)
    
    # Prepare for modeling
    X_train, y_train = feature_engineer.prepare_for_modeling(train_features)
    X_test, y_test = feature_engineer.prepare_for_modeling(test_features)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Total features: {len(feature_engineer.get_feature_columns())}")
    print(f"\nFeature list:")
    for i, feat in enumerate(feature_engineer.get_feature_columns(), 1):
        print(f"  {i}. {feat}")
    print("="*60)
