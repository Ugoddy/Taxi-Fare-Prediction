"""Shared feature code for the taxi fare project.

The notebook, the FastAPI app and the Streamlit app all import from this file,
so a trip is turned into model features exactly the same way everywhere.
Keep it next to the notebook (and later inside the app folder).
"""
from datetime import datetime

import numpy as np
import pandas as pd

# Columns the saved pipeline expects as input (same as test_fare.csv)
RAW_COLUMNS = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "passenger_count", "year", "month", "day", "dayOfWeek", "hour",
    "timeOfDayMid.Day", "timeOfDayMorning", "timeOfDayNight",
]

COORD_COLUMNS = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]

# Rough NYC bounding box used for cleaning and for checking app inputs
NYC_BOX = {"lon_min": -76.0, "lon_max": -72.0, "lat_min": 38.0, "lat_max": 42.0}

# The dataset codes years as 1-7 (1 = 2009) and weekdays as 1-7 (1 = Sunday).
# Both were checked against every row of train_fare.csv.
YEAR_OFFSET = 2008
FIRST_YEAR, LAST_YEAR = 2009, 2015

LANDMARKS = {
    "downtown": (40.7141667, -74.0063889),
    "jfk": (40.6441667, -73.7822222),
    "ewr": (40.69, -74.175),
    "lgr": (40.77, -73.87),
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_features(df):
    """Raw trip columns -> model features (distances in km, raw coordinates dropped).

    Used as the first step of the sklearn Pipeline, so the saved model accepts
    raw trips and does this itself.
    """
    df = pd.DataFrame(df, columns=RAW_COLUMNS).copy()
    lat1, lon1 = df["pickup_latitude"], df["pickup_longitude"]
    lat2, lon2 = df["dropoff_latitude"], df["dropoff_longitude"]

    df["trip_distance_km"] = haversine_km(lat1, lon1, lat2, lon2)
    for name, (lat, lon) in LANDMARKS.items():
        df[f"{name}_pickup_km"] = haversine_km(lat, lon, lat1, lon1)
        df[f"{name}_dropoff_km"] = haversine_km(lat, lon, lat2, lon2)

    return df.drop(columns=COORD_COLUMNS)


def time_of_day_flags(hour):
    """Same buckets as the dataset: morning 3-8, mid-day 9-13, night 18-2, afternoon 14-17 = all zeros."""
    return {
        "timeOfDayMid.Day": int(9 <= hour <= 13),
        "timeOfDayMorning": int(3 <= hour <= 8),
        "timeOfDayNight": int(hour >= 18 or hour <= 2),
    }


def trip_to_row(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat, passenger_count, pickup_datetime):
    """One trip as a user would describe it -> one-row DataFrame in RAW_COLUMNS order.

    This is what the FastAPI app will call.
    """
    if isinstance(pickup_datetime, str):
        pickup_datetime = datetime.fromisoformat(pickup_datetime)
    dt = pickup_datetime
    row = {
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "passenger_count": passenger_count,
        "year": dt.year - YEAR_OFFSET,
        "month": dt.month,
        "day": dt.day,
        "dayOfWeek": (dt.weekday() + 1) % 7 + 1,  # Python Monday=0 -> dataset Sunday=1
        "hour": dt.hour,
        **time_of_day_flags(dt.hour),
    }
    return pd.DataFrame([row], columns=RAW_COLUMNS)
