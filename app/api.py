import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Stock Prediction API",
    description="REST API for deep learning stock price prediction",
    version="1.0"
)

MODEL_PATH = "../models/task2_1_vietnam_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)


class StockInput(BaseModel):
    data: list


@app.get("/")
def home():
    return {
        "message": "Stock Prediction API is running"
    }


@app.post("/predict")
def predict(input_data: StockInput):
    arr = np.array(input_data.data, dtype=np.float32)

    # Expected shape: (1, 30, 5)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr)

    return {
        "prediction": prediction.tolist()
    }
