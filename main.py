"""FastAPI service for the NYC taxi fare model.

Run locally:   uvicorn main:app --reload
Then open:     http://127.0.0.1:8000/docs
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from taxi_features import FIRST_YEAR, LAST_YEAR, haversine_km, trip_to_row

BASE_DIR = Path(__file__).resolve().parent

# Load the model and its config once, when the app starts
model = joblib.load(BASE_DIR / "taxi_pipeline.joblib")
config = json.loads((BASE_DIR / "taxi_config.json").read_text())

LON_MIN, LON_MAX = config["valid_ranges"]["longitude"]
LAT_MIN, LAT_MAX = config["valid_ranges"]["latitude"]
MAX_TRAINING_FARE = config["training_fare_range"][1]

app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description=(
        "Predicts a New York taxi fare from pickup and dropoff locations, pickup time "
        "and number of passengers. Random Forest trained on 1M+ NYC trips (2009-2015), "
        f"average error ${config['validation_metrics']['MAE']:.2f}, "
        f"R² {config['validation_metrics']['R2']:.2f}."
    ),
    version="1.0.0",
)


class Trip(BaseModel):
    """One taxi trip, described the way a rider would."""

    pickup_longitude: float = Field(..., ge=LON_MIN, le=LON_MAX, examples=[-73.9855])
    pickup_latitude: float = Field(..., ge=LAT_MIN, le=LAT_MAX, examples=[40.7580])
    dropoff_longitude: float = Field(..., ge=LON_MIN, le=LON_MAX, examples=[-73.9772])
    dropoff_latitude: float = Field(..., ge=LAT_MIN, le=LAT_MAX, examples=[40.7527])
    passenger_count: int = Field(1, ge=1, le=6, examples=[1])
    pickup_datetime: datetime = Field(..., examples=["2013-03-05T09:30:00"])


class FarePrediction(BaseModel):
    predicted_fare: float
    currency: str = "USD"
    trip_distance_km: float
    note: Optional[str] = None


def check_trip(trip: Trip) -> None:
    """Reject trips the model was never trained on, with a clear reason."""
    if not FIRST_YEAR <= trip.pickup_datetime.year <= LAST_YEAR:
        raise HTTPException(
            status_code=422,
            detail=f"pickup_datetime must be between {FIRST_YEAR} and {LAST_YEAR}: "
                   "the model was trained on trips from those years only.",
        )
    if (trip.pickup_longitude == trip.dropoff_longitude
            and trip.pickup_latitude == trip.dropoff_latitude):
        raise HTTPException(status_code=422, detail="Pickup and dropoff are the same point.")


def predict_fare(trip: Trip) -> FarePrediction:
    """Core prediction logic, kept separate from the web layer so it is easy to test."""
    check_trip(trip)
    row = trip_to_row(
        trip.pickup_longitude, trip.pickup_latitude,
        trip.dropoff_longitude, trip.dropoff_latitude,
        trip.passenger_count, trip.pickup_datetime,
    )
    fare = float(model.predict(row)[0])
    distance = float(haversine_km(trip.pickup_latitude, trip.pickup_longitude,
                                  trip.dropoff_latitude, trip.dropoff_longitude))

    note = None
    if fare >= MAX_TRAINING_FARE - 2:
        note = (f"Close to the highest fare in the training data (${MAX_TRAINING_FARE:.2f}). "
                "The real fare for this trip may be higher.")

    return FarePrediction(predicted_fare=round(fare, 2), trip_distance_km=round(distance, 2), note=note)


@app.get("/")
def home():
    return {"message": "NYC Taxi Fare Prediction API. Go to /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {
        "model": config["model"],
        "sklearn_version": config["sklearn_version"],
        "validation_metrics": config["validation_metrics"],
        "baseline_MAE": config["baseline_MAE"],
        "training_fare_range": config["training_fare_range"],
        "valid_ranges": config["valid_ranges"],
    }


@app.post("/predict", response_model=FarePrediction)
def predict(trip: Trip):
    return predict_fare(trip)
