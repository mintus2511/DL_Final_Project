# Deep Learning for Artificial Intelligence Final Project

# Intelligent Stock Market Prediction, Trading Signal Identification, and Portfolio Optimization using Deep Learning

---

## Project Overview

This project was developed as the final project for the course **Deep Learning for Artificial Intelligence**. The project focuses on applying deep learning techniques to real-world financial market analysis using both Nasdaq and Vietnam stock market datasets.

The project investigates how deep learning models can be used to:

* Predict future stock prices
* Perform short-term and multi-step forecasting
* Identify buy and sell trading signals
* Construct risk-aware investment portfolios
* Deploy trained models as practical AI services

The project combines:

* Time-series forecasting
* Financial data preprocessing
* Deep learning experimentation
* Classification modeling
* Portfolio management logic
* Lightweight AI deployment

Multiple deep learning architectures were explored, including:

* Convolutional Neural Networks (CNN)
* Long Short-Term Memory Networks (LSTM)

The final system includes:

* forecasting notebooks,
* trading signal classifiers,
* portfolio optimization modules,
* FastAPI model deployment,
* and a Streamlit SaaS-style interface.

---

# Project Objectives

The main objectives of this project are:

1. Develop deep learning models capable of predicting future stock prices.
2. Compare CNN and LSTM architectures for financial time-series forecasting.
3. Identify trading opportunities through buy/sell signal classification.
4. Build a simplified portfolio management framework combining profitability and risk.
5. Deploy trained models using modern AI engineering tools.
6. Create an end-to-end practical AI pipeline for financial market analysis.

---

# Datasets

The project uses two different financial datasets.

## 1. Nasdaq Historical Stock Dataset

Used for:

* Task 1 forecasting experiments
* CNN vs LSTM comparison
* multi-step forecasting analysis

Features included:

* Open
* High
* Low
* Close
* Volume
* Adjusted Close

---

## 2. Vietnam Stock Market Dataset

Used for:

* Vietnam market forecasting
* trading signal identification
* portfolio optimization
* deployment demonstration

Selected Vietnam stocks:

* FPT
* VCB
* VNM
* HPG
* MBB

Features included:

* Open
* High
* Low
* Close
* Volume
* TradingDate

---

# Project Structure

```text
DL_Final_Project/
│
├── app/
│   ├── api.py
│   ├── streamlit_app.py
│   └── workflow_description.md
│
├── data/
│   ├── nasdaq/
│   └── vietnam/
│
├── models/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_task1_cnn_prediction.ipynb
│   ├── 03_lstm_prediction.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_task2_vietnam_prediction.ipynb
│   ├── 06_task3_trading_signal.ipynb
│   └── 07_task4_portfolio_risk.ipynb
│
├── results/
│
├── saved_objects/
│
├── report/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Methodology

## 1. Data Preprocessing

The preprocessing pipeline includes:

* missing value handling
* chronological sorting
* feature selection
* MinMax normalization
* time-window sequence generation

For forecasting tasks, a sliding window approach was used:

* Input window size: 30 trading days
* Output:

  * next-day prediction
  * kth-day prediction
  * multi-step prediction

---

## 2. Time-Series Splitting

The dataset was split chronologically to avoid data leakage.

Split ratio:

* Training set: 70%
* Validation set: 15%
* Test set: 15%

Random shuffling was intentionally avoided because stock market data is sequential.

---

## 3. Deep Learning Models

### CNN Architecture

The CNN forecasting architecture includes:

* Conv1D layers
* MaxPooling1D
* Dropout regularization
* Dense fully connected layers

CNN was selected because it can efficiently extract local temporal patterns from financial sequences.

---

### LSTM Architecture

LSTM models were implemented for comparison in:

* kth-day prediction
* multi-step forecasting

LSTM networks are designed to capture long-term temporal dependencies in sequential data.

---

## 4. Technical Indicators

For trading signal identification, several financial indicators were engineered:

* SMA (Simple Moving Average)
* EMA (Exponential Moving Average)
* MACD
* RSI
* Volatility
* Daily Return

These indicators help the model capture:

* momentum,
* trend direction,
* and market stability.

---

# Task 1 — Nasdaq Stock Forecasting

## Task Description

Task 1 focuses on forecasting Nasdaq stock prices using deep learning models.

Three forecasting scenarios were implemented:

1. Next-day prediction
2. kth-day prediction
3. Multi-step prediction

---

## Task 1.1 — Next-Day Prediction

A CNN model was trained to predict the next trading day closing price using a 30-day historical window.

### Result

CNN achieved strong performance with low forecasting error and stable prediction trends.

---

## Task 1.2 — kth-Day Prediction

The objective was to predict the stock price at a future kth day.

Both CNN and LSTM models were evaluated.

### Observation

LSTM slightly outperformed CNN in kth-day forecasting because recurrent networks capture longer temporal dependencies more effectively.

---

## Task 1.3 — Multi-Step Forecasting

The model predicted the next 7 consecutive trading days simultaneously.

### Observation

CNN performed more stably than LSTM in multi-step forecasting. LSTM accumulated forecasting error more rapidly over longer prediction horizons.

---

## Final Task 1 Model Selection

| Task     | Best Model |
| -------- | ---------- |
| Task 1.1 | CNN        |
| Task 1.2 | LSTM       |
| Task 1.3 | CNN        |

---

# Task 2 — Vietnam Stock Market Forecasting

## Task Description

Task 2 extends the forecasting pipeline to Vietnam stock market data.

The following multi-feature inputs were used:

* Low
* Open
* Volume
* High
* Close

---

## Task 2 Results

| Task                              | RMSE   | MAE    |
| --------------------------------- | ------ | ------ |
| Task 2.1 — Next-day prediction    | 0.0361 | 0.0276 |
| Task 2.2 — 7th day prediction     | 0.0494 | 0.0369 |
| Task 2.3 — Next 7 days prediction | 0.0730 | 0.0594 |

---

## Key Observations

* Forecasting error increased as prediction horizon increased.
* CNN generalized effectively on Vietnam stock market data.
* Vietnam stock prediction behaved differently from Nasdaq forecasting experiments.
* Multi-step forecasting remained significantly more difficult than next-day prediction.

---

# Task 3 — Trading Signal Identification

## Task Description

This task reformulates stock analysis as a classification problem.

Instead of predicting prices directly, the model predicts:

* Buy signals
* Sell signals

based on future return thresholds.

---

## Class Imbalance Problem

Initial experiments suffered from severe class imbalance because large market movements were relatively rare.

The models initially predicted only the majority class.

To address this issue:

* signal thresholds were adjusted,
* class weighting was introduced,
* and classification evaluation metrics were expanded.

---

## Task 3.1 — Buy Signal Identification

### Final Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.5050 |
| Precision | 0.4885 |
| Recall    | 0.9717 |
| F1-score  | 0.6501 |

### Interpretation

The buy signal model achieved very strong recall, meaning it successfully captured most potential buying opportunities. However, the model generated many false positives, which is common in noisy financial classification tasks.

---

## Task 3.2 — Sell Signal Identification

### Final Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.6672 |
| Precision | 0.3056 |
| Recall    | 0.3077 |
| F1-score  | 0.3066 |

### Interpretation

Sell signal prediction was more difficult because strong downward movements were relatively rare in the dataset. Nevertheless, the model successfully learned both classes after threshold adjustment and class weighting.

---

# Task 4 — Portfolio Composition and Risk Management

## Task Description

Task 4 combines profitability analysis and risk management to build a simplified investment portfolio.

The portfolio framework evaluates:

* expected profitability,
* volatility,
* maximum drawdown,
* and risk-adjusted allocation.

---

## Profitability Analysis

Profitability was evaluated using:

* cumulative return over 120 trading days,
* average daily return.

### Key Findings

| Stock | Observation                        |
| ----- | ---------------------------------- |
| VCB   | Highest profitability              |
| VNM   | Most stable stock                  |
| HPG   | High volatility and strong decline |
| MBB   | Weak recent performance            |

---

## Risk Analysis

Risk was measured using:

* volatility,
* maximum drawdown.

VNM demonstrated:

* the lowest volatility,
* the smallest drawdown,
* and the most defensive characteristics.

---

## Final Portfolio Allocation

| Stock | Allocation |
| ----- | ---------- |
| VNM   | 39.93%     |
| VCB   | 35.49%     |
| FPT   | 24.57%     |

---

## Portfolio Interpretation

* VNM received the largest allocation because of its low-risk defensive behavior.
* VCB was selected as a strong growth-oriented component.
* FPT provided diversification and technology sector exposure.

---

# Task 5 — Deployment and AI Engineering Workflow

## FastAPI Deployment

A lightweight REST API was developed using FastAPI.

The API:

* loads trained TensorFlow models,
* receives JSON requests,
* returns prediction outputs.

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## Streamlit SaaS Application

A Streamlit web application was implemented to simulate a lightweight SaaS interface.

Features include:

* interactive prediction requests,
* configurable inputs,
* prediction visualization,
* model interpretation,
* API communication.

---

## AI Workflow Design

The proposed AI engineering workflow includes:

* data ingestion,
* preprocessing,
* model training,
* model serving,
* and frontend interaction.

The workflow demonstrates how deep learning systems can be integrated into practical AI applications.

---

# Environment Setup

## 1. Create Virtual Environment

```bash
python -m venv venv
```

---

## 2. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

## Run FastAPI Backend

```bash
cd app
python -m uvicorn api:app --reload
```

---

## Run Streamlit Frontend

```bash
cd app
python -m streamlit run streamlit_app.py
```

---

# Key Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* FastAPI
* Streamlit

---

# Limitations

Several limitations remain in this project:

* Financial markets are highly noisy and difficult to predict.
* Only historical market features were used.
* Macroeconomic indicators and news sentiment were excluded.
* Portfolio optimization was simplified.
* Real trading constraints such as transaction cost were not modeled.

---

# Future Improvements

Future extensions may include:

* Transformer-based forecasting models
* Reinforcement learning trading agents
* Real-time data streaming
* Advanced portfolio optimization algorithms
* Docker and cloud deployment
* Real-time market dashboard systems

---

# Conclusion

This project demonstrates how deep learning can be applied to financial market analysis through forecasting, classification, portfolio management, and deployment.

CNN and LSTM architectures successfully captured temporal market patterns under multiple forecasting scenarios. Trading signal identification demonstrated the challenges of class imbalance in financial classification tasks, while portfolio optimization illustrated the importance of balancing profitability and risk.

Finally, the deployment phase showed how deep learning models can be integrated into lightweight production-style AI systems using FastAPI and Streamlit.

Overall, the project combines theoretical deep learning concepts with practical AI engineering workflows in a real-world financial context.

---

# References

* TensorFlow Documentation
* Keras Documentation
* Scikit-learn Documentation
* FastAPI Documentation
* Streamlit Documentation
* Yahoo Finance Dataset
* Vietnam Stock Historical Dataset
