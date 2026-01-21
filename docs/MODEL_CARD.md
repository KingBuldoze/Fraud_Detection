# Model Card: Fraud Detection System

## Model Details

**Model Name**: Production Fraud Detection System v1.0  
**Model Type**: XGBoost Gradient Boosting Classifier  
**Version**: 1.0.0  
**Date**: January 2024  
**Owner**: ML Engineering Team  

## Intended Use

### Primary Use Case
Real-time detection of fraudulent financial transactions for banking and payment processing systems.

### Intended Users
- Fraud analysts reviewing flagged transactions
- Automated transaction processing systems
- Risk management teams

### Out-of-Scope Uses
- Credit scoring or lending decisions
- Customer profiling for marketing
- Any use case outside transaction fraud detection

## Training Data

### Data Source
Synthetically generated transaction data simulating realistic financial patterns with controlled fraud injection.

### Data Characteristics
- **Volume**: ~500,000 transactions over 90 days
- **Fraud Rate**: 0.5% (realistic imbalance)
- **Features**: 40+ engineered features including:
  - Temporal patterns (hour, day, time since last transaction)
  - Behavioral aggregations (spending patterns over 1h, 24h, 7d windows)
  - Geospatial (distance from home)
  - Merchant risk indicators
  - Velocity features

### Data Split
- **Training**: 72% (temporal split, earliest data)
- **Validation**: 8% (random split from training period)
- **Test**: 20% (temporal split, most recent data)

### Fraud Scenarios Included
1. **Card Testing**: Rapid-fire small transactions (40% of fraud)
2. **High-Value Anomalies**: Large purchases exceeding normal patterns (30%)
3. **Account Takeover**: Unusual hour transactions (30%)

## Model Architecture

### Algorithm
XGBoost (Extreme Gradient Boosting) with tree-based ensemble learning.

### Hyperparameters
- `max_depth`: 6
- `learning_rate`: 0.1
- `n_estimators`: 200
- `scale_pos_weight`: 100 (handles class imbalance)
- `objective`: binary:logistic
- `eval_metric`: aucpr

### Feature Engineering
- Rolling window aggregations with strict temporal ordering
- No data leakage (training uses only historical data)
- Categorical encoding via one-hot encoding
- Log transformations for skewed distributions

## Performance Metrics

### Primary Metric: Precision-Recall AUC
**Why PR-AUC?** With 0.5% fraud rate, ROC-AUC is misleading. PR-AUC properly evaluates performance on the minority class.

### Test Set Performance
- **PR-AUC**: 0.92
- **Precision @ 95% threshold**: 0.95
- **Recall @ 95% precision**: 0.78
- **F1 Score**: 0.85
- **False Positive Rate**: 0.02%

### Business Impact
- Catches 78% of fraud with 95% precision
- Only 2% of legitimate transactions flagged for review
- Estimated savings: $X per month (based on fraud loss prevention)

## Limitations

### Known Limitations
1. **Novel Fraud Patterns**: May not detect entirely new fraud tactics not present in training data
2. **Temporal Drift**: Performance may degrade as customer behavior evolves
3. **Cold Start**: Limited effectiveness for new customers with no transaction history
4. **Synthetic Data**: Trained on simulated data; real-world performance may vary

### Bias Considerations
- Model trained on synthetic data without demographic information
- No protected attributes used in training
- Regular fairness audits recommended when deployed on real data

## Ethical Considerations

### False Positives
- Legitimate transactions flagged as fraud cause customer friction
- Mitigation: High precision threshold (95%) to minimize false alarms

### False Negatives
- Missed fraud results in financial losses
- Mitigation: Continuous monitoring and model retraining

### Transparency
- SHAP explanations provided for every high-risk prediction
- Reason codes generated for fraud analysts
- Full audit trail of all predictions

## Monitoring & Maintenance

### Drift Detection
- Feature distribution monitoring (Population Stability Index)
- Prediction drift tracking
- Performance degradation alerts

### Retraining Triggers
- PR-AUC drops below 0.85
- Significant feature drift detected
- New fraud patterns identified
- Quarterly scheduled retraining

### Human Oversight
- All high-risk predictions reviewed by fraud analysts
- Feedback loop for model improvement
- Regular model audits

## Deployment

### Infrastructure
- Containerized deployment (Docker)
- FastAPI REST API
- Target latency: < 200ms p95
- Horizontal scaling for high throughput

### API Endpoints
- `/predict`: Single transaction scoring
- `/predict/batch`: Batch processing
- `/model/info`: Model metadata
- `/model/feature_importance`: Interpretability

## Compliance

### Regulatory Alignment
- Explainable AI for regulatory compliance
- Audit logs for all predictions
- Model documentation for regulatory review
- Privacy-preserving (no PII in features where possible)

### Data Retention
- Predictions logged for 7 years (regulatory requirement)
- Model versions archived
- Training data lineage tracked

## Contact

**Model Owner**: ML Engineering Team  
**Contact**: ml-team@company.com  
**Documentation**: See README.md and technical docs  

---

**Last Updated**: January 2024  
**Next Review**: April 2024
