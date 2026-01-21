# Deployment Guide

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- 4GB RAM minimum
- 2 CPU cores minimum

## Local Development Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd fraud-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Training Data
```bash
python -m fraud_detection.data.generator
```

This creates `data/transactions.csv` with ~500K transactions.

### 5. Train Model
```bash
python -m fraud_detection.models.train
```

Training takes ~5-10 minutes. Model saved to `models/fraud_model_<timestamp>.joblib`.

### 6. Run API Server
```bash
uvicorn fraud_detection.api.app:app --reload --port 8000
```

API available at `http://localhost:8000`

### 7. Test API
```bash
# Health check
curl http://localhost:8000/

# Predict fraud
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_transaction.json
```

## Docker Deployment

### Build Image
```bash
docker build -t fraud-detection:latest .
```

### Run Container
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --name fraud-api \
  fraud-detection:latest
```

### Check Logs
```bash
docker logs -f fraud-api
```

## Cloud Deployment

### AWS (ECS/Fargate)

1. **Push to ECR**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag fraud-detection:latest <account>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
```

2. **Create ECS Task Definition**
- Container: fraud-detection
- Port: 8000
- CPU: 1024
- Memory: 2048
- Health check: `/`

3. **Create ECS Service**
- Launch type: Fargate
- Desired tasks: 2 (for HA)
- Load balancer: Application Load Balancer

### GCP (Cloud Run)

```bash
gcloud builds submit --tag gcr.io/<project-id>/fraud-detection
gcloud run deploy fraud-detection \
  --image gcr.io/<project-id>/fraud-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure (Container Instances)

```bash
az container create \
  --resource-group fraud-detection-rg \
  --name fraud-api \
  --image <registry>.azurecr.io/fraud-detection:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000
```

## Production Considerations

### Environment Variables
```bash
export FRAUD_MODEL_PATH=/app/models/fraud_model.joblib
export LOG_LEVEL=INFO
export MAX_WORKERS=4
```

### Scaling
- **Horizontal**: Add more container instances behind load balancer
- **Vertical**: Increase CPU/memory for higher throughput
- **Target**: 100 requests/second per instance

### Monitoring
- **Metrics**: Latency, throughput, error rate
- **Logging**: Structured JSON logs to CloudWatch/Stackdriver
- **Alerts**: 
  - Latency > 200ms
  - Error rate > 1%
  - Model drift detected

### Security
- **API Authentication**: Add API key or OAuth
- **TLS**: Terminate TLS at load balancer
- **Network**: Deploy in private VPC
- **Secrets**: Use AWS Secrets Manager / GCP Secret Manager

### Model Updates
1. Train new model
2. Save with timestamp
3. Update model path in config
4. Rolling deployment (blue-green)
5. Monitor performance
6. Rollback if degraded

## Testing

### Unit Tests
```bash
pytest tests/test_features.py -v
```

### Integration Tests
```bash
pytest tests/test_api.py -v
```

### Load Testing
```bash
pip install locust
locust -f tests/load_test.py --host http://localhost:8000
```

## Troubleshooting

### Model Not Loading
- Check model file exists in `models/` directory
- Verify file permissions
- Check logs for errors

### High Latency
- Check feature computation time
- Verify database/feature store connection
- Consider caching frequent customers

### Low Accuracy
- Check for data drift
- Verify feature engineering consistency
- Retrain model with recent data

## Maintenance

### Daily
- Monitor API health and latency
- Check error logs

### Weekly
- Review drift detection metrics
- Analyze false positive/negative rates

### Monthly
- Retrain model with new data
- Update feature engineering if needed
- Review model performance trends

### Quarterly
- Full model audit
- Update documentation
- Security review
