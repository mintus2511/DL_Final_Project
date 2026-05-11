import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Vietnam Stock Prediction SaaS",
    layout="wide"
)

st.title("Vietnam Stock Price Prediction SaaS")
st.write(
    """
    This demo is a lightweight SaaS-style application for the Deep Learning final project.
    It connects to a FastAPI model service and returns a predicted normalized closing price
    based on a 30-day stock input window.
    """
)

st.info(
    """
    Model used: CNN stock prediction model trained on Vietnam stock historical data.
    Input shape expected by the API: 30 days × 5 features.
    Features: Low, Open, Volume, High, Close.
    """
)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Model Settings")

api_url = st.sidebar.text_input(
    "API endpoint",
    value="http://127.0.0.1:8000/predict"
)

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="FPT"
)

window_size = 30
num_features = 5

st.sidebar.write("Window size:", window_size)
st.sidebar.write("Number of features:", num_features)

input_mode = st.sidebar.radio(
    "Input mode",
    [
        "Demo generated data",
        "Manual normalized values"
    ]
)

# ============================================================
# Demo generated input
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
        close_series + np.random.normal(0, noise, window_size), 0, 1)
    high_series = np.clip(close_series + 0.03, 0, 1)
    low_series = np.clip(close_series - 0.03, 0, 1)
    volume_series = np.clip(np.random.normal(0.5, 0.1, window_size), 0, 1)

    input_df = pd.DataFrame({
        "Low": low_series,
        "Open": open_series,
        "Volume": volume_series,
        "High": high_series,
        "Close": close_series
    })

else:
    st.subheader("Manual Normalized Input")

    st.write(
        """
        Enter one normalized value for each feature.
        The app will repeat this value for all 30 days to create a model input window.
        """
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        low = st.number_input("Low", 0.0, 1.0, 0.45)

    with col2:
        open_price = st.number_input("Open", 0.0, 1.0, 0.50)

    with col3:
        volume = st.number_input("Volume", 0.0, 1.0, 0.50)

    with col4:
        high = st.number_input("High", 0.0, 1.0, 0.55)

    with col5:
        close = st.number_input("Close", 0.0, 1.0, 0.50)

    input_df = pd.DataFrame(
        np.tile(
            [low, open_price, volume, high, close],
            (window_size, 1)
        ),
        columns=["Low", "Open", "Volume", "High", "Close"]
    )

# ============================================================
# Display input
# ============================================================

st.subheader("Input Data Preview")

st.dataframe(input_df.head(10), use_container_width=True)

st.subheader("30-Day Normalized Close Price Pattern")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(input_df["Close"], marker="o")
ax.set_xlabel("Day")
ax.set_ylabel("Normalized Close Price")
ax.set_title(f"{ticker} - Input Close Price Pattern")
st.pyplot(fig)

# ============================================================
# Prediction
# ============================================================

st.subheader("Model Prediction")

if st.button("Run Prediction"):

    sample_input = input_df.values.tolist()

    try:
        response = requests.post(
            api_url,
            json={"data": sample_input},
            timeout=10
        )

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            predicted_value = prediction[0][0]

            st.success("Prediction completed successfully.")

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    label="Predicted Normalized Close Price",
                    value=f"{predicted_value:.4f}"
                )

            with metric_col2:
                last_close = input_df["Close"].iloc[-1]
                change = predicted_value - last_close

                st.metric(
                    label="Predicted Change vs Last Close",
                    value=f"{change:.4f}"
                )

            if predicted_value > last_close:
                st.write(
                    "Model interpretation: predicted price is higher than the latest input close price.")
                st.warning("Possible signal: upward movement.")
            elif predicted_value < last_close:
                st.write(
                    "Model interpretation: predicted price is lower than the latest input close price.")
                st.warning("Possible signal: downward movement.")
            else:
                st.write(
                    "Model interpretation: predicted price is almost unchanged.")

        else:
            st.error(
                "Prediction failed. Please check whether the FastAPI server is running.")

    except Exception as e:
        st.error("Could not connect to the API server.")
        st.write(e)

# ============================================================
# Methodology Section
# ============================================================

st.divider()

st.subheader("Project Methodology")

st.write(
    """
    This application demonstrates the deployment stage of the Deep Learning for Artificial Intelligence final project.
    The stock prediction model was trained using historical Vietnam stock market data. A 30-day time window was used
    as input, and the model predicted the next normalized closing price.
    """
)

st.markdown(
    """
    **Pipeline summary:**

    1. Load historical stock price data.
    2. Clean and sort data chronologically.
    3. Select multi-feature inputs: Low, Open, Volume, High, Close.
    4. Normalize features using MinMaxScaler.
    5. Create 30-day time windows.
    6. Train a 1D CNN model.
    7. Save the trained model.
    8. Serve the model using FastAPI.
    9. Build this Streamlit interface as a lightweight SaaS demo.
    """
)

st.subheader("Important Notes")

st.write(
    """
    The output is a normalized prediction, not a direct investment recommendation.
    In a real financial system, this prediction should be combined with technical indicators,
    risk scoring, portfolio constraints, and human decision-making.
    """
)
