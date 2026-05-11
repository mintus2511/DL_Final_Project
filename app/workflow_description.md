# AI Engineering Workflow for Stock Price Prediction

The proposed workflow automates the stock prediction pipeline from data ingestion to model serving.

## 1. Data Ingestion

Airbyte is used to pull stock market data from external sources or internal CSV storage into a centralized database.

## 2. Data Storage

PostgreSQL stores raw historical price data, processed feature tables, model predictions, and portfolio outputs.

## 3. Data Transformation

dbt transforms raw price data into model-ready feature tables. Features include daily return, moving averages, volatility, RSI, and MACD.

## 4. Model Training and Prediction

A scheduled Python training script loads processed features, trains the deep learning model, evaluates performance, and saves the best model.

## 5. Workflow Orchestration

Apache Airflow schedules the full pipeline daily:
- ingest new data
- clean and transform data
- run model prediction
- store prediction results
- update dashboard data

## 6. Model Serving

FastAPI exposes the trained TensorFlow model through a REST endpoint. Other systems can call the API to receive predictions.

## 7. User Interface

Streamlit provides a lightweight SaaS interface where users can input recent stock data and view model predictions.

## 8. Monitoring

Prediction results, model metrics, and portfolio outputs are stored for later evaluation and visualization.