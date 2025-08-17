[Go to the 2nd attempt](https://github.com/Chrmorod/Project_Final_Stock_Prices_Prediction) 
# 📈 Stock Price Prediction Pipeline

This project implements a modular and scalable architecture for **30-day stock price prediction**, using various modeling techniques and modern tools for orchestration, traceability, and visualization.

## 🚀 Quick Start

```bash
STOCK=AAPL YEAR_BACK=5 ./start.sh
```

> Runs the pipeline for ticker `AAPL` using data from the past 5 years.

---

## 📐 Architecture

The architecture consists of the following components:

1. **Python**: modular scripts for data collection, preprocessing, and modeling.
2. **Mage.ai**: orchestrator for workflows (ETL + ML).
3. **MLflow**: for tracking, logging, and comparing models.
4. **Grafana**: dashboards for visualization of predictions and metrics.
5. **Models**:
   - **Monte Carlo** simulation
   - **ARIMA** (AutoRegressive Integrated Moving Average)
   - **LSTM** (Long Short-Term Memory Neural Network)

---

## 🧪 Step-by-Step Pipeline

### 1. Data Ingestion (Mage)

- Connects to **Yahoo Finance** to download historical stock data.
- Configurable via `.env` or CLI parameters (`STOCK`, `YEAR_BACK`).
- Daily frequency.

### 2. Preprocessing (Mage)

- Null cleaning, adjustment for splits/dividends.
- Normalization for neural network models.
- Stationarity detection (for ARIMA).

### 3. Modeling (Python + MLflow)

Each model is trained in its own Mage node:

#### 🔁 Monte Carlo

- Generates 10,000 simulated paths.
- Uses historical log return mean and standard deviation.

#### 📊 ARIMA

- (p,d,q) parameters selected via AIC/BIC.
- Residual diagnostics for validation.

#### 🔮 LSTM

- Keras sequential architecture with 1-2 LSTM + Dense layers.
- Trained using sliding window technique.

**MLflow Tracking:**
- Artifacts: serialized models (`.pkl`, `.h5`), plots, logs.
- Model version comparison in MLflow UI.

### 4. Visualization (Grafana)

- Exports predictions and metrics to InfluxDB or Prometheus.
- Dynamic Grafana dashboard includes:
  - Real vs. predicted time series.
  - Error distribution per model.
  - Model performance ranking.
---

## 🧠 Technical Requirements

- Python ≥ 3.9
- Mage.ai
- MLflow
- TensorFlow / Keras
- statsmodels, yfinance, scikit-learn
- Docker (optional)
- Grafana

Quick install:

```bash
pip install -r requirements.txt
```
---

## 📊 Example Dashboard in Grafana

The dashboard includes:

- 📈 Predicted prices & Average predicted prices
---

## 🛠️ Customization

Easily adapt the pipeline:

- Change ticker or time window in `start.sh`.
- Tune model hyperparameters in their Mage nodes.
- Add new models (Prophet, XGBoost, etc.) by duplicating and adapting nodes.
---

## 📦 Deployment
Option: Docker

```bash
./start.sh
```

Includes containers for:
- Docker 
- Mage
- MLflow Tracking Server
- Grafana
- Sqlite
- Postgres

---
