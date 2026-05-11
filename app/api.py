from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf


# ============================================================
# Load trained model
# ============================================================

MODEL_PATH = "../models/task2_vietnam_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Vietnam Stock Price Prediction API",
    description="REST API for serving a trained CNN stock price prediction model.",
    version="1.0"
)


# ============================================================
# Input schema
# ============================================================

class StockInput(BaseModel):
    data: List[List[float]]


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Vietnam Stock Price Prediction API is running.",
        "model": "CNN stock prediction model",
        "expected_input_shape": "30 days x 5 features"
    }


# ============================================================
# Health check endpoint
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(input_data: StockInput):
    try:
        data = np.array(input_data.data, dtype=np.float32)

        if data.shape != (30, 5):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input shape {data.shape}. Expected shape is (30, 5)."
            )

        data = np.expand_dims(data, axis=0)

        prediction = model.predict(data)

        return {
            "prediction": prediction.tolist(),
            "input_shape": list(data.shape),
            "message": "Prediction completed successfully."
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
