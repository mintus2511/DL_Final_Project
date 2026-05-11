import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Vietnam Stock Prediction SaaS",
    layout="wide"
)

# ============================================================
# Header
# ============================================================

st.title("Vietnam Stock Price Prediction SaaS")

st.markdown(
    """
    This application demonstrates a lightweight SaaS-style deployment for the
    Deep Learning final project. It connects a Streamlit frontend with a FastAPI
    backend and serves a trained CNN model for Vietnam stock price prediction.
    """
)

st.info(
    """
    **Model:** CNN stock prediction model  
    **Input shape:** 30 days × 5 features  
    **Features:** Low, Open, Volume, High, Close  
    **Output:** Predicted normalized closing price
    """
)

# ============================================================
# Sidebar Settings
# ============================================================

st.sidebar.header("Application Settings")

api_url = st.sidebar.text_input(
    "FastAPI Prediction Endpoint",
    value="http://127.0.0.1:8000/predict"
)

ticker = st.sidebar.selectbox(
    "Select Stock Ticker",
    ["FPT", "VCB", "VNM", "HPG", "MBB"]
)

input_mode = st.sidebar.radio(
    "Input Mode",
    [
        "Demo generated data",
        "Manual normalized values"
    ]
)

window_size = 30
num_features = 5

st.sidebar.markdown("---")
st.sidebar.write("Window size:", window_size)
st.sidebar.write("Number of features:", num_features)

# ============================================================
# API Health Check
# ============================================================

st.subheader("API Service Status")

health_url = api_url.replace("/predict", "/health")

try:
    health_response = requests.get(
        health_url,
        timeout=3
    )

    if health_response.status_code == 200:
        st.success("FastAPI backend is running successfully.")
    else:
        st.warning(
            "FastAPI backend responded, but the health check was not successful."
        )

except Exception:
    st.error(
        "FastAPI backend is not connected. Please run the API server first."
    )

# ============================================================
# Input Data Generator
# ============================================================

if input_mode == "Demo generated data":

    st.subheader("Demo Input Generator")

    col1, col2, col3 = st.columns(3)

    with col1:
        base_value = st.slider(
            "Base normalized price level",
            0.0,
            1.0,
            0.5
        )

    with col2:
        trend = st.slider(
            "Trend strength",
            -0.05,
            0.05,
            0.01
        )

    with col3:
        noise = st.slider(
            "Noise level",
            0.0,
            0.1,
            0.02
        )

    np.random.seed(42)

    days = np.arange(window_size)

    close_series = base_value + trend * days + np.random.normal(
        0,
        noise,
        window_size
    )

    close_series = np.clip(close_series, 0, 1)

    open_series = np.clip(
        close_series + np.random.normal(0, noise, window_size),
        0,
        1
    )

    high_series = np.clip(
        close_series + 0.03,
        0,
        1
    )

    low_series = np.clip(
        close_series - 0.03,
        0,
        1
    )

    volume_series = np.clip(
        np.random.normal(0.5, 0.1, window_size),
        0,
        1
    )

    input_df = pd.DataFrame({
        "Low": low_series,
        "Open": open_series,
        "Volume": volume_series,
        "High": high_series,
        "Close": close_series
    })

else:

    st.subheader("Manual Normalized Input")

    st.markdown(
        """
        Enter normalized base values for each feature.

        Instead of repeating identical values for all 30 days,
        the app now generates a small realistic variation pattern
        to simulate actual market movement.
        """
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        low = st.number_input(
            "Low",
            min_value=0.0,
            max_value=1.0,
            value=0.45
        )

    with col2:
        open_price = st.number_input(
            "Open",
            min_value=0.0,
            max_value=1.0,
            value=0.50
        )

    with col3:
        volume = st.number_input(
            "Volume",
            min_value=0.0,
            max_value=1.0,
            value=0.50
        )

    with col4:
        high = st.number_input(
            "High",
            min_value=0.0,
            max_value=1.0,
            value=0.55
        )

    with col5:
        close = st.number_input(
            "Close",
            min_value=0.0,
            max_value=1.0,
            value=0.50
        )

    # ========================================================
    # Create realistic 30-day variation
    # ========================================================

    days = np.arange(window_size)

    low_series = low + np.sin(days / 5) * 0.01

    open_series = open_price + np.cos(days / 6) * 0.01

    volume_series = volume + np.sin(days / 7) * 0.02

    high_series = high + np.cos(days / 4) * 0.01

    close_series = close + np.sin(days / 3) * 0.01

    # ========================================================
    # Combine all features
    # ========================================================

    manual_data = np.column_stack([
        low_series,
        open_series,
        volume_series,
        high_series,
        close_series
    ])

    # Keep values between 0 and 1
    manual_data = np.clip(manual_data, 0, 1)

    input_df = pd.DataFrame(
        manual_data,
        columns=[
            "Low",
            "Open",
            "Volume",
            "High",
            "Close"
        ]
    )

# ============================================================
# Input Preview
# ============================================================

st.subheader("Input Data Preview")

st.dataframe(
    input_df,
    use_container_width=True
)

st.subheader("30-Day Input Close Price Pattern")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    input_df["Close"],
    marker="o",
    label="Input Close Price"
)

ax.set_title(f"{ticker} - 30-Day Normalized Close Price Input")
ax.set_xlabel("Day")
ax.set_ylabel("Normalized Close Price")
ax.legend()

st.pyplot(fig)

# ============================================================
# Prediction Section
# ============================================================

st.subheader("Model Prediction")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if st.button("Run Prediction"):

    sample_input = input_df.values.tolist()

    try:
        response = requests.post(
            api_url,
            json={
                "data": sample_input
            },
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]

            predicted_value = float(prediction[0][0])

            last_close = float(input_df["Close"].iloc[-1])

            change = predicted_value - last_close

            st.success("Prediction completed successfully.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted Normalized Close Price",
                    f"{predicted_value:.4f}"
                )

            with col2:
                st.metric(
                    "Latest Input Close",
                    f"{last_close:.4f}"
                )

            with col3:
                st.metric(
                    "Predicted Change",
                    f"{change:.4f}"
                )

            st.subheader("Prediction Interpretation")

            if change > 0.03:
                st.success(
                    "Model interpretation: strong upward movement may occur."
                )
            elif change > 0:
                st.info(
                    "Model interpretation: slight upward movement may occur."
                )
            elif change < -0.03:
                st.warning(
                    "Model interpretation: strong downward movement may occur."
                )
            elif change < 0:
                st.info(
                    "Model interpretation: slight downward movement may occur."
                )
            else:
                st.info(
                    "Model interpretation: price is expected to remain almost unchanged."
                )

            # ========================================================
            # Prediction Visualization
            # ========================================================

            st.subheader("Prediction Visualization")

            fig2, ax2 = plt.subplots(figsize=(10, 4))

            ax2.plot(
                input_df["Close"].values,
                label="Historical Input Close"
            )

            ax2.scatter(
                len(input_df),
                predicted_value,
                label="Predicted Next Close",
                marker="o",
                s=100
            )

            ax2.plot(
                [len(input_df) - 1, len(input_df)],
                [last_close, predicted_value],
                linestyle="--",
                label="Predicted Movement"
            )

            ax2.set_title(f"{ticker} - Prediction Output")
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Normalized Close Price")
            ax2.legend()

            st.pyplot(fig2)

            # ========================================================
            # Save Prediction History
            # ========================================================

            st.session_state.prediction_history.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Ticker": ticker,
                "Last_Close": round(last_close, 4),
                "Predicted_Close": round(predicted_value, 4),
                "Predicted_Change": round(change, 4),
                "API_Status": response.status_code
            })

        else:
            st.error(
                "Prediction failed. Please check the FastAPI server or input format."
            )
            st.write(response.text)

    except Exception as e:
        st.error("Could not connect to the API server.")
        st.write(e)

# ============================================================
# Prediction History
# ============================================================

st.subheader("Prediction History")

if len(st.session_state.prediction_history) > 0:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv = history_df.to_csv(index=False)

    st.download_button(
        label="Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

else:
    st.write("No predictions have been made yet.")

# ============================================================
# SaaS Architecture
# ============================================================

st.divider()

st.subheader("SaaS Deployment Architecture")

st.markdown(
    """
    This application demonstrates a lightweight Software-as-a-Service (SaaS)
    architecture for deploying deep learning models in financial forecasting.

    The system separates the frontend user interface from the backend prediction
    engine using REST API communication. This architecture follows modern AI
    engineering practices commonly used in production environments.

    ### System Components

    #### 1. Streamlit Frontend
    The Streamlit application provides the interactive web interface for users.
    It allows users to:

    - select Vietnam stock tickers,
    - generate or manually create normalized stock inputs,
    - visualize historical input patterns,
    - request predictions from the backend model,
    - and monitor prediction history.

    The frontend focuses on usability and rapid interaction.

    #### 2. FastAPI Backend
    FastAPI serves as the prediction API layer.

    The backend:
    - loads the trained CNN model,
    - receives JSON requests from Streamlit,
    - validates input shapes,
    - performs inference,
    - and returns prediction outputs as JSON responses.

    FastAPI was selected because it is lightweight, fast, and widely used for
    machine learning deployment.

    #### 3. Trained Deep Learning Model
    The deployed model is a 1D CNN trained on Vietnam stock market data.

    The model learns temporal patterns from:
    - Low price,
    - Open price,
    - Volume,
    - High price,
    - and Close price.

    The model input shape is:
    - 30 trading days
    - × 5 normalized features

    The output is a normalized predicted closing price.

    #### 4. REST API Communication
    Communication between Streamlit and FastAPI uses HTTP REST requests.

    Streamlit sends input sequences in JSON format:

    ```json
    {
        "data": [[...], [...], ...]
    }
    ```

    The FastAPI backend returns predictions as JSON responses:

    ```json
    {
        "prediction": [[0.4888]]
    }
    ```

    This design makes the system modular and scalable.

    #### 5. Prediction Visualization Layer
    After receiving predictions from the API,
    Streamlit visualizes:
    - input close-price history,
    - predicted next-day movement,
    - prediction direction,
    - and prediction history tracking.

    This improves interpretability and usability.

    ### End-to-End Workflow

    ```text
    User
      ↓
    Streamlit Web Interface
      ↓
    Input Processing & Visualization
      ↓
    FastAPI REST API
      ↓
    Trained CNN Model
      ↓
    Prediction Output
      ↓
    Streamlit Dashboard Visualization
      ↓
    User Decision Support
    ```

    ### Engineering Benefits

    This architecture demonstrates several industry-relevant AI engineering concepts:

    - model serving,
    - API-based deployment,
    - frontend-backend separation,
    - reproducible inference,
    - lightweight SaaS deployment,
    - and interactive ML applications.

    The system can later be extended into:
    - cloud deployment,
    - real-time stock prediction,
    - multi-model serving,
    - automated retraining,
    - or portfolio recommendation systems.
    """
)

# ============================================================
# Methodology
# ============================================================

st.subheader("Project Methodology")

st.markdown(
    """
    The deployed model was trained using a complete deep learning
    time-series forecasting pipeline.

    ### Step 1 — Data Collection

    Historical Vietnam stock market data was collected for multiple companies.

    The dataset contains:
    - Open price,
    - High price,
    - Low price,
    - Close price,
    - Volume,
    - and trading dates.

    ### Step 2 — Data Cleaning and Preprocessing

    The preprocessing stage included:
    - handling missing values,
    - chronological sorting,
    - feature selection,
    - normalization using MinMaxScaler,
    - and sliding-window sequence generation.

    Normalization was important because deep learning models train more
    effectively when features are scaled consistently.

    ### Step 3 — Sliding Window Construction

    A 30-day rolling window approach was used.

    This means:
    - the previous 30 trading days
    - are used to predict
    - the next future closing price.

    This transforms raw time-series data into supervised learning samples.

    ### Step 4 — CNN Model Training

    A 1D Convolutional Neural Network (CNN) was trained.

    CNN layers help capture:
    - short-term temporal movement,
    - local price patterns,
    - and feature interactions.

    The model was trained using:
    - Mean Squared Error (MSE) loss,
    - Adam optimizer,
    - and chronological train-validation-test splitting.

    ### Step 5 — Model Saving

    After training, the final model was saved using TensorFlow `.keras` format.

    This allows:
    - reproducibility,
    - deployment,
    - and inference reuse without retraining.

    ### Step 6 — API Deployment

    The trained model was deployed using FastAPI.

    The API:
    - loads the saved model,
    - accepts input requests,
    - performs inference,
    - and returns prediction outputs.

    Swagger UI documentation is automatically generated for API testing.

    ### Step 7 — Streamlit SaaS Integration

    The Streamlit frontend connects directly to the FastAPI backend.

    Users can:
    - create input sequences,
    - request predictions,
    - visualize outputs,
    - and interact with the model through a web interface.

    ### Step 8 — Prediction Monitoring

    The application stores prediction history during runtime,
    enabling lightweight monitoring and export functionality.

    ### Final Outcome

    The final system demonstrates an end-to-end AI workflow:
    - data preprocessing,
    - deep learning training,
    - model evaluation,
    - API deployment,
    - SaaS frontend integration,
    - and interactive prediction serving.
    """
)
# ============================================================
# AI Automation Workflow
# ============================================================

st.divider()

st.subheader("AI Automation Workflow")

st.markdown(
    """
    This project follows a simplified AI engineering workflow inspired by
    real-world machine learning deployment systems.

    The workflow automates the movement from raw stock market data
    to deployable prediction services.

    ### Workflow Stages

    #### 1. Data Collection
    Historical stock market data is collected from CSV datasets containing:
    - Open price,
    - High price,
    - Low price,
    - Close price,
    - Volume,
    - and trading dates.

    #### 2. Data Preprocessing
    The raw financial data is:
    - cleaned,
    - sorted chronologically,
    - normalized,
    - and transformed into supervised learning sequences.

    #### 3. Feature Engineering
    Multiple financial indicators can be generated, including:
    - SMA,
    - EMA,
    - RSI,
    - MACD,
    - Volatility,
    - and Daily Returns.

    These engineered features improve the model's ability to learn
    market behavior.

    #### 4. Deep Learning Training
    CNN and LSTM models are trained using:
    - sliding window sequences,
    - chronological validation,
    - and forecasting/classification objectives.

    #### 5. Model Evaluation
    The models are evaluated using:
    - RMSE,
    - MAE,
    - Accuracy,
    - Precision,
    - Recall,
    - F1-score,
    - and confusion matrices.

    #### 6. Model Serialization
    Trained models are saved in TensorFlow `.keras` format
    for deployment and reproducibility.

    #### 7. API Deployment
    FastAPI loads the trained model and exposes prediction endpoints
    through REST APIs.

    #### 8. SaaS Frontend Deployment
    Streamlit provides the user-facing dashboard for:
    - prediction requests,
    - visualization,
    - and interactive financial analysis.

    #### 9. User Prediction Workflow
    Users interact with the Streamlit interface,
    which automatically communicates with the FastAPI backend
    to generate prediction outputs.

    ### End-to-End Automation Pipeline

    ```text
    Raw Stock Data
          ↓
    Data Cleaning
          ↓
    Feature Engineering
          ↓
    Normalization
          ↓
    Sliding Window Creation
          ↓
    CNN / LSTM Training
          ↓
    Model Evaluation
          ↓
    Saved .keras Model
          ↓
    FastAPI REST API
          ↓
    Streamlit SaaS Dashboard
          ↓
    User Prediction
    ```

    ### Engineering Concepts Demonstrated

    This project demonstrates several important AI engineering principles:

    - reproducible ML workflows,
    - model deployment,
    - API serving,
    - frontend-backend separation,
    - lightweight SaaS architecture,
    - and end-to-end ML system integration.

    Although simplified for academic purposes,
    the workflow structure follows many concepts used in
    production AI systems.
    """
)

# ============================================================
# Important Notes
# ============================================================

st.subheader("Important Notes")

st.warning(
    """
    This application is designed for academic demonstration and educational purposes only.

    The prediction outputs are normalized model estimates generated from historical stock data.
    They should not be interpreted as guaranteed investment signals or financial advice.

    Financial markets are highly volatile and influenced by many external factors,
    including:
    - macroeconomic conditions,
    - company announcements,
    - geopolitical events,
    - interest rates,
    - market sentiment,
    - and unexpected news.

    The deployed model only uses historical price-based features and does not incorporate:
    - real-time news,
    - financial statements,
    - order-book information,
    - or advanced quantitative indicators.

    Therefore, prediction accuracy is inherently limited.

    In real-world production systems, additional components would normally be required:
    - risk management systems,
    - portfolio optimization,
    - transaction cost analysis,
    - real-time streaming pipelines,
    - model monitoring,
    - automated retraining,
    - cloud infrastructure,
    - authentication and security,
    - and human financial oversight.

    This project demonstrates the integration of deep learning,
    API deployment, and interactive SaaS engineering workflows
    within a practical stock forecasting application.
    """
)
