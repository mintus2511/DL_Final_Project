# AI Engineering Workflow for Intelligent Stock Market Forecasting

This document describes the proposed AI engineering workflow for the stock market forecasting and portfolio analysis system developed in this project.

The workflow demonstrates how deep learning models can be integrated into a simplified end-to-end machine learning pipeline inspired by real-world AI production systems.

The architecture combines:
- automated data ingestion,
- data transformation,
- deep learning model training,
- workflow orchestration,
- model serving,
- and interactive SaaS deployment.

---

# Overall Workflow Architecture

```text
External Stock Data Sources
            ↓
        Airbyte
            ↓
       PostgreSQL
            ↓
            dbt
            ↓
    Feature Engineering
            ↓
      Model Training
            ↓
      Model Evaluation
            ↓
      Saved .keras Model
            ↓
         FastAPI
            ↓
      Streamlit SaaS
            ↓
      User Prediction
            ↓
      Monitoring Layer
```

---

# 1. Data Ingestion Layer

The workflow begins with financial market data ingestion.

Historical stock market data can be collected from:
- Yahoo Finance,
- Vietnam stock market datasets,
- CSV storage systems,
- or external financial APIs.

Airbyte is proposed as the ingestion tool because it supports automated extraction and synchronization of data from multiple external sources into centralized storage systems.

The ingestion layer is responsible for:
- pulling new stock market records,
- synchronizing updated datasets,
- and maintaining structured historical data.

This design supports scalability and automation for future real-time deployment systems.

---

# 2. Centralized Data Storage

PostgreSQL is proposed as the primary database system.

The database stores:
- raw historical stock prices,
- processed feature tables,
- engineered technical indicators,
- model prediction outputs,
- portfolio analysis results,
- and monitoring information.

A centralized database improves:
- reproducibility,
- query efficiency,
- model tracking,
- and deployment organization.

The database layer also allows future integration with:
- dashboards,
- automated retraining systems,
- and real-time streaming applications.

---

# 3. Data Transformation and Feature Engineering

After ingestion, raw financial data is transformed into model-ready datasets.

dbt (Data Build Tool) is proposed for data transformation because it enables:
- modular SQL transformations,
- version-controlled feature pipelines,
- reproducible preprocessing workflows,
- and maintainable feature engineering systems.

The transformation pipeline includes:
- chronological sorting,
- missing value handling,
- normalization,
- rolling-window generation,
- and technical indicator engineering.

Generated technical indicators may include:
- SMA (Simple Moving Average),
- EMA (Exponential Moving Average),
- RSI (Relative Strength Index),
- MACD,
- volatility,
- and daily returns.

These engineered features help the deep learning models learn:
- market momentum,
- trend direction,
- temporal behavior,
- and market stability.

---

# 4. Deep Learning Training Pipeline

The model training stage loads processed feature tables from the transformed datasets.

The project implements:
- CNN forecasting models,
- LSTM forecasting models,
- and CNN-based trading signal classifiers.

The training pipeline performs:
- chronological train-validation-test splitting,
- sliding-window sequence generation,
- hyperparameter experimentation,
- and model evaluation.

The models are evaluated using:
- RMSE,
- MAE,
- Accuracy,
- Precision,
- Recall,
- F1-score,
- and confusion matrices.

The best-performing models are saved using TensorFlow `.keras` format for deployment and reproducibility.

---

# 5. Workflow Orchestration

Apache Airflow is proposed as the workflow orchestration system.

Airflow automates the scheduling and execution of the full AI pipeline.

The orchestration workflow may include:
- ingesting new market data,
- running preprocessing tasks,
- generating technical indicators,
- retraining models,
- evaluating performance,
- storing predictions,
- and refreshing deployment outputs.

Example scheduled workflow:

1. Pull new stock data
2. Update PostgreSQL tables
3. Run dbt transformations
4. Generate forecasting features
5. Execute model prediction
6. Save prediction outputs
7. Refresh dashboard information

This design improves:
- automation,
- reproducibility,
- scalability,
- and maintainability.

---

# 6. Model Serving Layer

FastAPI is used as the model serving framework.

The FastAPI service:
- loads trained TensorFlow models,
- receives JSON prediction requests,
- validates input data,
- performs inference,
- and returns prediction outputs through REST APIs.

The API exposes:
- prediction endpoints,
- health-check endpoints,
- and Swagger documentation.

Example deployment endpoints:

```text
http://127.0.0.1:8000/predict
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

FastAPI was selected because it is:
- lightweight,
- high-performance,
- easy to integrate,
- and widely adopted in machine learning deployment systems.

---

# 7. SaaS Frontend Interface

Streamlit is used as the lightweight SaaS-style frontend application.

The Streamlit application allows users to:
- select Vietnam stock tickers,
- generate or input stock sequences,
- visualize historical stock movement,
- request predictions,
- monitor prediction history,
- and analyze portfolio outputs.

The frontend communicates directly with the FastAPI backend through REST API requests.

This frontend-backend separation follows modern AI engineering deployment practices and improves modularity.

---

# 8. Monitoring and Tracking Layer

The monitoring layer stores:
- prediction outputs,
- evaluation metrics,
- model performance summaries,
- and portfolio allocation results.

Monitoring supports:
- model evaluation,
- prediction consistency analysis,
- and future retraining decisions.

A future production-level system could extend monitoring to include:
- drift detection,
- real-time alert systems,
- automated retraining triggers,
- and model version management.

---

# Engineering Concepts Demonstrated

This workflow demonstrates several important AI engineering principles:

- end-to-end ML pipelines,
- reproducible preprocessing,
- automated workflow orchestration,
- API-based model serving,
- frontend-backend separation,
- SaaS deployment architecture,
- and practical machine learning system integration.

Although simplified for academic purposes, the proposed architecture resembles many concepts used in modern production AI systems.

---

# Future Extensions

Future improvements may include:
- real-time market streaming,
- Docker containerization,
- cloud deployment,
- CI/CD integration,
- Kubernetes orchestration,
- automated retraining pipelines,
- reinforcement learning portfolio optimization,
- and large-scale distributed forecasting systems.

The workflow can also be extended into a fully automated MLOps pipeline for large-scale financial prediction systems.