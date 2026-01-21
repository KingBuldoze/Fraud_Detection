# System Architecture

## Overview

The fraud detection system is designed as a microservices architecture with clear separation between training and inference pipelines.

```
┌─────────────────────────────────────────────────────────────┐
│                     FRAUD DETECTION SYSTEM                   │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│  Data Generation │────────▶│  Feature Store   │
│   (Synthetic)    │         │   (Historical)   │
└──────────────────┘         └──────────────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │ Feature Engineer │
                             │  (Aggregations)  │
                             └──────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌──────────────────┐              ┌──────────────────┐
          │  Model Training  │              │  Real-time API   │
          │    (XGBoost)     │              │    (FastAPI)     │
          └──────────────────┘              └──────────────────┘
                    │                                   │
                    ▼                                   ▼
          ┌──────────────────┐              ┌──────────────────┐
          │ Model Registry   │─────────────▶│   Predictions    │
          │  (Versioned)     │              │   (Logged)       │
          └──────────────────┘              └──────────────────┘
                                                      │
                                                      ▼
                                            ┌──────────────────┐
                                            │ Drift Detection  │
                                            │   & Monitoring   │
                                            └──────────────────┘
```

## Components

### 1. Data Layer

#### Data Generator (`fraud_detection/data/generator.py`)
- **Purpose**: Simulate realistic transaction data
- **Features**:
  - Customer profiles with spending patterns
  - Terminal/merchant locations
  - Temporal transaction patterns
  - Fraud scenario injection (3 types)
- **Output**: CSV file with ~500K transactions

#### Feature Store (Conceptual)
- In production: Redis/DynamoDB for real-time features
- Stores customer aggregations (1h, 24h, 7d windows)
- Enables low-latency feature retrieval

### 2. Feature Engineering Layer

#### Feature Engineer (`fraud_detection/features/pipeline.py`)
- **Purpose**: Transform raw transactions into ML features
- **Key Features**:
  - Temporal: hour, day, weekend indicator
  - Behavioral: spending patterns, transaction frequency
  - Velocity: time since last transaction
  - Risk: terminal fraud rate, distance from home
- **Leak Prevention**: Training mode uses only historical data

### 3. Model Layer

#### Training Pipeline (`fraud_detection/models/train.py`)
- **Algorithm**: XGBoost with class imbalance handling
- **Evaluation**: PR-AUC (primary), ROC-AUC, precision/recall
- **Threshold Optimization**: Cost-based or precision-targeted
- **Output**: Versioned model artifacts

#### Explainability (`fraud_detection/models/explainability.py`)
- **Method**: SHAP (SHapley Additive exPlanations)
- **Outputs**:
  - Global feature importance
  - Local prediction explanations
  - Human-readable reason codes

### 4. Inference Layer

#### FastAPI Service (`fraud_detection/api/app.py`)
- **Endpoints**:
  - `POST /predict`: Single transaction scoring
  - `POST /predict/batch`: Batch processing
  - `GET /model/info`: Model metadata
  - `GET /model/feature_importance`: Interpretability
- **Features**:
  - Request validation (Pydantic)
  - Latency monitoring
  - Audit logging
  - Error handling

### 5. Monitoring Layer

#### Drift Detection (`fraud_detection/monitoring/drift_detection.py`)
- **Feature Drift**: Statistical tests on feature distributions
- **Prediction Drift**: KS test on prediction distributions
- **Performance Monitoring**: Track PR-AUC over time
- **Alerts**: Trigger retraining when drift detected

## Data Flow

### Training Pipeline
```
Raw Data → Feature Engineering → Train/Val/Test Split → 
Model Training → Threshold Optimization → Model Evaluation → 
Model Registry
```

### Inference Pipeline
```
Transaction Request → Feature Engineering → Model Prediction → 
SHAP Explanation → Response (fraud_probability + reason_codes) → 
Audit Log
```

## Design Decisions

### Why XGBoost?
- Excellent performance on tabular data
- Handles missing values natively
- Built-in feature importance
- Fast inference (< 10ms)
- SHAP support for explainability

### Why PR-AUC over ROC-AUC?
- Fraud rate is 0.5% (highly imbalanced)
- ROC-AUC optimistic on imbalanced data
- PR-AUC focuses on minority class performance
- Industry standard for fraud detection

### Why Temporal Split?
- Simulates production scenario (train on past, predict future)
- Prevents data leakage
- Tests model's ability to generalize to new patterns

### Why SHAP?
- Model-agnostic explanations
- Theoretically grounded (Shapley values)
- Regulatory compliance (explainable AI)
- Generates actionable reason codes

## Scalability

### Current Capacity
- **Throughput**: ~100 requests/second per instance
- **Latency**: p95 < 150ms
- **Model Size**: ~50MB
- **Memory**: ~2GB per instance

### Scaling Strategy
- **Horizontal**: Add API instances behind load balancer
- **Caching**: Cache frequent customer features
- **Batch Processing**: Use `/predict/batch` for bulk scoring
- **Feature Store**: Pre-compute aggregations

## Security

### Data Security
- No PII in logs
- Encrypted data at rest
- TLS for data in transit

### API Security
- API key authentication
- Rate limiting
- Input validation
- CORS configuration

### Model Security
- Model versioning and rollback
- Audit trail for predictions
- Access control for model updates

## Compliance

### Regulatory Requirements
- **Explainability**: SHAP explanations for every prediction
- **Audit Trail**: All predictions logged with timestamps
- **Model Documentation**: Model card with performance metrics
- **Bias Monitoring**: Regular fairness audits

### Data Governance
- Data retention policies (7 years)
- Model lineage tracking
- Version control for code and models

## Future Enhancements

### Short-term
1. Graph-based features (transaction networks)
2. Deep learning for sequential patterns (LSTM)
3. AutoML for hyperparameter tuning
4. A/B testing framework

### Long-term
1. Federated learning (privacy-preserving)
2. Real-time feature store (Feast/Tecton)
3. Multi-model ensemble
4. Automated retraining pipeline
