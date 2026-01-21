"""
Realistic transaction data generator with fraud pattern injection.

This module simulates a financial transaction dataset with:
- Customer profiles with spending patterns
- Terminal/merchant locations
- Temporal transaction patterns (time of day, day of week)
- Realistic fraud scenarios injected at controlled rates
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
from faker import Faker
from pathlib import Path

from fraud_detection.utils import Config, setup_logger

logger = setup_logger(__name__)


class CustomerProfile:
    """Represents a customer with spending behavior"""
    
    def __init__(self, customer_id: int, fake: Faker):
        self.customer_id = customer_id
        self.name = fake.name()
        
        # Spending profile - log-normal distribution
        self.avg_transaction_amount = np.random.lognormal(mean=3.5, sigma=0.8)  # ~$50 avg
        self.transaction_std = self.avg_transaction_amount * 0.4
        
        # Frequency profile (transactions per day)
        self.avg_daily_transactions = np.random.poisson(lam=2) + 1
        
        # Preferred transaction hours (normal around peak hours)
        self.preferred_hour = int(np.random.normal(loc=14, scale=4) % 24)
        
        # Geographic location (for distance-based features)
        self.home_lat = fake.latitude()
        self.home_lon = fake.longitude()
        
        # Risk profile (some customers more susceptible to fraud)
        self.risk_score = np.random.beta(a=2, b=5)  # Skewed toward low risk


class Terminal:
    """Represents a merchant terminal/location"""
    
    def __init__(self, terminal_id: int, fake: Faker):
        self.terminal_id = terminal_id
        self.merchant_name = fake.company()
        self.merchant_category = np.random.choice([
            'grocery', 'restaurant', 'gas_station', 'online_retail',
            'electronics', 'clothing', 'entertainment', 'travel', 'other'
        ])
        
        # Location
        self.lat = fake.latitude()
        self.lon = fake.longitude()
        
        # Average transaction amount at this terminal
        self.avg_amount = np.random.lognormal(mean=3.0, sigma=1.0)


class TransactionGenerator:
    """Generate realistic transaction dataset with fraud patterns"""
    
    def __init__(
        self,
        num_customers: int = 10000,
        num_terminals: int = 1000,
        simulation_days: int = 90,
        fraud_ratio: float = 0.005,
        random_state: int = 42
    ):
        self.num_customers = num_customers
        self.num_terminals = num_terminals
        self.simulation_days = simulation_days
        self.fraud_ratio = fraud_ratio
        
        np.random.seed(random_state)
        self.fake = Faker()
        Faker.seed(random_state)
        
        # Generate profiles
        logger.info(f"Generating {num_customers} customer profiles...")
        self.customers = [CustomerProfile(i, self.fake) for i in range(num_customers)]
        
        logger.info(f"Generating {num_terminals} terminal profiles...")
        self.terminals = [Terminal(i, self.fake) for i in range(num_terminals)]
        
        self.start_date = datetime(2024, 1, 1)
        self.end_date = self.start_date + timedelta(days=simulation_days)
    
    def generate_legitimate_transaction(
        self,
        customer: CustomerProfile,
        timestamp: datetime
    ) -> Dict:
        """Generate a single legitimate transaction"""
        
        # Select terminal (customers have preferred terminals)
        terminal = np.random.choice(self.terminals)
        
        # Amount based on customer and terminal profiles
        amount = max(
            1.0,
            np.random.normal(
                loc=(customer.avg_transaction_amount + terminal.avg_amount) / 2,
                scale=customer.transaction_std
            )
        )
        
        # Distance from home (most transactions are local)
        distance_from_home = self._calculate_distance(
            customer.home_lat, customer.home_lon,
            terminal.lat, terminal.lon
        )
        
        return {
            'transaction_id': None,  # Will be assigned later
            'timestamp': timestamp,
            'customer_id': customer.customer_id,
            'terminal_id': terminal.terminal_id,
            'amount': round(amount, 2),
            'merchant_category': terminal.merchant_category,
            'distance_from_home': round(distance_from_home, 2),
            'is_fraud': 0
        }
    
    def inject_fraud_scenario_1(
        self,
        customer: CustomerProfile,
        timestamp: datetime
    ) -> List[Dict]:
        """
        Fraud Scenario 1: Rapid-fire small transactions (card testing)
        Fraudster tests stolen card with multiple small purchases
        """
        transactions = []
        num_attempts = np.random.randint(5, 15)
        
        for i in range(num_attempts):
            terminal = np.random.choice(self.terminals)
            # Small amounts for testing
            amount = np.random.uniform(1, 10)
            
            # Very short time intervals
            tx_time = timestamp + timedelta(minutes=i * np.random.randint(1, 5))
            
            # Random locations (not near home)
            distance = np.random.uniform(50, 500)
            
            transactions.append({
                'transaction_id': None,
                'timestamp': tx_time,
                'customer_id': customer.customer_id,
                'terminal_id': terminal.terminal_id,
                'amount': round(amount, 2),
                'merchant_category': terminal.merchant_category,
                'distance_from_home': round(distance, 2),
                'is_fraud': 1
            })
        
        return transactions
    
    def inject_fraud_scenario_2(
        self,
        customer: CustomerProfile,
        timestamp: datetime
    ) -> Dict:
        """
        Fraud Scenario 2: High-value anomalous transaction
        Single large purchase way above customer's normal pattern
        """
        terminal = np.random.choice(self.terminals)
        
        # Amount 5-10x customer's average
        amount = customer.avg_transaction_amount * np.random.uniform(5, 10)
        
        # Often in high-risk categories
        high_risk_categories = ['electronics', 'online_retail', 'travel']
        merchant_category = np.random.choice(high_risk_categories)
        
        # Often far from home
        distance = np.random.uniform(100, 1000)
        
        return {
            'transaction_id': None,
            'timestamp': timestamp,
            'customer_id': customer.customer_id,
            'terminal_id': terminal.terminal_id,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'distance_from_home': round(distance, 2),
            'is_fraud': 1
        }
    
    def inject_fraud_scenario_3(
        self,
        customer: CustomerProfile,
        timestamp: datetime
    ) -> List[Dict]:
        """
        Fraud Scenario 3: Multiple transactions at unusual hours
        Account takeover with transactions at odd hours
        """
        transactions = []
        num_transactions = np.random.randint(3, 8)
        
        # Unusual hours (late night / early morning)
        unusual_hours = [0, 1, 2, 3, 4, 5, 23]
        
        for i in range(num_transactions):
            terminal = np.random.choice(self.terminals)
            amount = np.random.uniform(50, 300)
            
            # Set to unusual hour
            tx_time = timestamp.replace(hour=np.random.choice(unusual_hours))
            tx_time += timedelta(minutes=i * np.random.randint(10, 30))
            
            distance = np.random.uniform(20, 200)
            
            transactions.append({
                'transaction_id': None,
                'timestamp': tx_time,
                'customer_id': customer.customer_id,
                'terminal_id': terminal.terminal_id,
                'amount': round(amount, 2),
                'merchant_category': terminal.merchant_category,
                'distance_from_home': round(distance, 2),
                'is_fraud': 1
            })
        
        return transactions
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete transaction dataset with fraud"""
        
        logger.info("Generating transaction dataset...")
        
        transactions = []
        current_date = self.start_date
        
        # Track customers already used for fraud to avoid duplicates
        fraud_customers = set()
        
        while current_date < self.end_date:
            # Generate legitimate transactions for this day
            for customer in self.customers:
                # Number of transactions for this customer today
                num_tx = np.random.poisson(customer.avg_daily_transactions)
                
                for _ in range(num_tx):
                    # Random time during the day (biased toward preferred hour)
                    hour = int(np.random.normal(customer.preferred_hour, 3) % 24)
                    minute = np.random.randint(0, 60)
                    timestamp = current_date.replace(hour=hour, minute=minute)
                    
                    tx = self.generate_legitimate_transaction(customer, timestamp)
                    transactions.append(tx)
            
            current_date += timedelta(days=1)
        
        logger.info(f"Generated {len(transactions)} legitimate transactions")
        
        # Inject fraud transactions
        num_fraud_needed = int(len(transactions) * self.fraud_ratio / (1 - self.fraud_ratio))
        logger.info(f"Injecting {num_fraud_needed} fraud transactions...")
        
        fraud_count = 0
        while fraud_count < num_fraud_needed:
            # Select random customer and timestamp
            customer = np.random.choice(self.customers)
            
            # Skip if already used for fraud (to spread fraud across customers)
            if customer.customer_id in fraud_customers and len(fraud_customers) < self.num_customers * 0.1:
                continue
            
            fraud_customers.add(customer.customer_id)
            
            # Random date during simulation period
            random_day = np.random.randint(0, self.simulation_days)
            timestamp = self.start_date + timedelta(days=random_day)
            
            # Select fraud scenario
            scenario = np.random.choice([1, 2, 3], p=[0.4, 0.3, 0.3])
            
            if scenario == 1:
                fraud_txs = self.inject_fraud_scenario_1(customer, timestamp)
                transactions.extend(fraud_txs)
                fraud_count += len(fraud_txs)
            elif scenario == 2:
                fraud_tx = self.inject_fraud_scenario_2(customer, timestamp)
                transactions.append(fraud_tx)
                fraud_count += 1
            else:
                fraud_txs = self.inject_fraud_scenario_3(customer, timestamp)
                transactions.extend(fraud_txs)
                fraud_count += len(fraud_txs)
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Sort by timestamp and assign transaction IDs
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['transaction_id'] = ['TX' + str(i).zfill(8) for i in range(len(df))]
        
        # Add temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Dataset generated: {len(df)} total transactions")
        logger.info(f"Fraud ratio: {df['is_fraud'].mean():.4f}")
        
        return df
    
    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate approximate distance in km using Haversine formula"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c


def main():
    """Generate and save transaction dataset"""
    
    Config.ensure_directories()
    
    generator = TransactionGenerator(
        num_customers=Config.NUM_CUSTOMERS,
        num_terminals=Config.NUM_TERMINALS,
        simulation_days=Config.SIMULATION_DAYS,
        fraud_ratio=Config.FRAUD_RATIO,
        random_state=Config.RANDOM_STATE
    )
    
    df = generator.generate_dataset()
    
    # Save to CSV
    output_path = Config.get_data_path("transactions.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total transactions: {len(df):,}")
    print(f"Fraud transactions: {df['is_fraud'].sum():,}")
    print(f"Fraud ratio: {df['is_fraud'].mean():.4%}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique customers: {df['customer_id'].nunique():,}")
    print(f"Unique terminals: {df['terminal_id'].nunique():,}")
    print(f"\nAmount statistics:")
    print(df.groupby('is_fraud')['amount'].describe())
    print("="*60)


if __name__ == "__main__":
    main()
