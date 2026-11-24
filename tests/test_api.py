"""Tests for the taxi fare API. Run with:  pytest -v"""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# Times Square -> Grand Central area, a short Midtown hop
SHORT_TRIP = {
    "pickup_longitude": -73.9855, "pickup_latitude": 40.7580,
    "dropoff_longitude": -73.9772, "dropoff_latitude": 40.7527,
    "passenger_count": 1, "pickup_datetime": "2013-03-05T09:30:00",
}

# Times Square -> JFK airport
LONG_TRIP = {**SHORT_TRIP, "dropoff_longitude": -73.7781, "dropoff_latitude": 40.6413}


def test_home():
    assert client.get("/").status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_has_metrics():
    body = client.get("/model-info").json()
    assert "MAE" in body["validation_metrics"]


def test_predict_short_trip():
    response = client.post("/predict", json=SHORT_TRIP)
    assert response.status_code == 200
    body = response.json()
    assert 3 <= body["predicted_fare"] <= 15
    assert body["currency"] == "USD"
    assert 0.5 <= body["trip_distance_km"] <= 1.5


def test_longer_trip_costs_more():
    short = client.post("/predict", json=SHORT_TRIP).json()["predicted_fare"]
    long = client.post("/predict", json=LONG_TRIP).json()["predicted_fare"]
    assert long > short


def test_airport_trip_gets_cap_note():
    body = client.post("/predict", json=LONG_TRIP).json()
    assert body["note"] is not None


def test_rejects_zero_passengers():
    assert client.post("/predict", json={**SHORT_TRIP, "passenger_count": 0}).status_code == 422


def test_rejects_location_outside_nyc():
    # London
    bad = {**SHORT_TRIP, "pickup_longitude": -0.1276, "pickup_latitude": 51.5072}
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_year_outside_training_data():
    assert client.post("/predict", json={**SHORT_TRIP, "pickup_datetime": "2024-01-01T10:00:00"}).status_code == 422


def test_rejects_same_pickup_and_dropoff():
    same = {**SHORT_TRIP, "dropoff_longitude": -73.9855, "dropoff_latitude": 40.7580}
    assert client.post("/predict", json=same).status_code == 422


def test_rejects_missing_field():
    incomplete = {k: v for k, v in SHORT_TRIP.items() if k != "pickup_latitude"}
    assert client.post("/predict", json=incomplete).status_code == 422
