# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.10+
- 4GB RAM
- 10GB disk space

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data (~2 minutes)
python -m fraud_detection.data.generator

# 3. Train model (~5-10 minutes)
python -m fraud_detection.models.train

# 4. Start API server
uvicorn fraud_detection.api.app:app --reload --port 8000
```

### Test the API

```bash
# Health check
curl http://localhost:8000/

# Predict fraud
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_transaction.json
```

### Expected Output

```json
{
  "transaction_id": "TX00123456",
  "is_fraud": false,
  "fraud_probability": 0.0234,
  "risk_level": "low",
  "reason_codes": [],
  "processing_time_ms": 45.2,
  "timestamp": "2024-01-15T14:30:05"
}
```

## 📊 What You Get

- **500K+ realistic transactions** with fraud patterns
- **XGBoost model** with 92% PR-AUC
- **Real-time API** with < 150ms latency
- **SHAP explanations** for every prediction
- **Drift detection** and monitoring
- **Complete test suite**
- **Docker deployment** ready

## 📚 Documentation

- [README.md](README.md) - Project overview
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) - Model documentation
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment guide
- [notebooks/01_exploratory_analysis.ipynb](notebooks/01_exploratory_analysis.ipynb) - EDA

## 🧪 Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_api.py -v
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t fraud-detection:latest .

# Run container
docker run -d -p 8000:8000 --name fraud-api fraud-detection:latest

# Check logs
docker logs -f fraud-api
```

## 🎯 Key Features

✅ **Production-Ready**: Real-world fraud scenarios, not toy data  
✅ **Explainable AI**: SHAP reason codes for compliance  
✅ **Proper Metrics**: PR-AUC for imbalanced data  
✅ **No Data Leakage**: Temporal validation  
✅ **Scalable**: Containerized, cloud-ready  
✅ **Monitored**: Drift detection built-in  
✅ **Tested**: Comprehensive test coverage  
✅ **Documented**: Enterprise-level docs  

## 💡 Next Steps

1. **Explore the code**: Start with `fraud_detection/data/generator.py`
2. **Run the notebook**: See `notebooks/01_exploratory_analysis.ipynb`
3. **Test the API**: Use the sample transaction
4. **Read the docs**: Understand the architecture
5. **Deploy**: Use Docker for production

## 🆘 Troubleshooting

**Model not loading?**
- Ensure you ran `python -m fraud_detection.models.train` first
- Check `models/` directory exists

**API errors?**
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check port 8000 is available

**Slow training?**
- Reduce dataset size in `fraud_detection/utils/config.py`
- Adjust `NUM_CUSTOMERS` and `SIMULATION_DAYS`

## 📧 Support

For questions or issues, see the documentation in `docs/`.

---

**Built with ❤️ for production fraud detection**
