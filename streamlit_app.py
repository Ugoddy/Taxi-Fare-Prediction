"""Streamlit front end for the NYC Taxi Fare Prediction API.

Start the API first (in its own window):
    python -m uvicorn main:app --reload --port 8001
Then run this app (in a second window):
    python -m streamlit run streamlit_app.py --server.port 8502
"""
import os
from datetime import date, datetime, time

import pandas as pd
import requests
import streamlit as st

# Where the FastAPI service is running. Docker/AWS will override this with an environment variable.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

# Well-known NYC places (latitude, longitude) so users don't need to type coordinates
PLACES = {
    "Times Square": (40.7580, -73.9855),
    "Grand Central Terminal": (40.7527, -73.9772),
    "Empire State Building": (40.7484, -73.9857),
    "Penn Station": (40.7506, -73.9935),
    "Central Park (Columbus Circle)": (40.7681, -73.9819),
    "Wall Street": (40.7060, -74.0088),
    "Brooklyn Bridge (Manhattan side)": (40.7061, -73.9969),
    "Williamsburg, Brooklyn": (40.7081, -73.9571),
    "Yankee Stadium, Bronx": (40.8296, -73.9262),
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Newark Airport": (40.6895, -74.1745),
}
CUSTOM = "Custom coordinates"

st.set_page_config(page_title="NYC Taxi Fare Predictor", page_icon="🚕", layout="centered")


@st.cache_data(ttl=60)
def _fetch_model_info():
    # Only successful answers are cached; a failure raises, so the next page load tries again.
    response = requests.get(f"{API_URL}/model-info", timeout=5)
    response.raise_for_status()
    return response.json()


def get_model_info():
    try:
        return _fetch_model_info()
    except requests.RequestException:
        return None


def location_picker(label, default_place, key):
    names = list(PLACES) + [CUSTOM]
    choice = st.selectbox(label, names, index=names.index(default_place), key=key)
    if choice == CUSTOM:
        lat = st.number_input("Latitude", min_value=40.4, max_value=41.0, value=40.7500,
                              step=0.001, format="%.4f", key=f"{key}_lat")
        lon = st.number_input("Longitude", min_value=-74.3, max_value=-73.6, value=-73.9800,
                              step=0.001, format="%.4f", key=f"{key}_lon")
        return lat, lon
    return PLACES[choice]


def error_message(response):
    """Turn an API error into a readable sentence."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return f"The API returned an error (code {response.status_code})."
    if isinstance(detail, list):  # pydantic validation errors
        return "; ".join(f"{'/'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
    return str(detail)


# ---------- Sidebar: about the model ----------
info = get_model_info()
with st.sidebar:
    st.header("About the model")
    if info:
        m = info["validation_metrics"]
        st.metric("Average error (MAE)", f"${m['MAE']:.2f}",
                  delta=f"-${info['baseline_MAE'] - m['MAE']:.2f} vs guessing the average",
                  delta_color="inverse")
        st.metric("R²", f"{m['R2']:.2f}")
        st.caption("Random Forest trained on 1M+ NYC taxi trips (2009–2015), "
                   "tested on 103,315 trips it never saw.")
        low, high = info["training_fare_range"]
        st.caption(f"Training fares range from ${low:.2f} to ${high:.2f}, "
                   "so the model can't predict fares above about $30.")
    else:
        st.error(f"Can't reach the API at {API_URL}. Start it with:\n\n"
                 "`python -m uvicorn main:app --port 8001`")
    st.markdown("[Code on GitHub](https://github.com/Ugoddy/Taxi-Fare-Prediction)")

# ---------- Main page ----------
st.title("🚕 NYC Taxi Fare Predictor")
st.write("Pick where the trip starts and ends, when it happens and how many passengers. "
         "The model estimates the fare before the trip starts.")

col1, col2 = st.columns(2)
with col1:
    pickup = location_picker("Pickup", "Times Square", key="pickup")
with col2:
    dropoff = location_picker("Dropoff", "Grand Central Terminal", key="dropoff")

col3, col4, col5 = st.columns(3)
with col3:
    trip_date = st.date_input("Pickup date", value=date(2014, 6, 13),
                              min_value=date(2009, 1, 1), max_value=date(2015, 12, 31),
                              help="The model learned from trips between 2009 and 2015.")
with col4:
    trip_time = st.time_input("Pickup time", value=time(18, 0))
with col5:
    passengers = st.slider("Passengers", min_value=1, max_value=6, value=1)

st.map(pd.DataFrame({"lat": [pickup[0], dropoff[0]], "lon": [pickup[1], dropoff[1]]}), zoom=10)

if st.button("Predict fare", type="primary"):
    if pickup == dropoff:
        st.warning("Pickup and dropoff are the same place. Choose two different points.")
    else:
        payload = {
            "pickup_latitude": pickup[0], "pickup_longitude": pickup[1],
            "dropoff_latitude": dropoff[0], "dropoff_longitude": dropoff[1],
            "passenger_count": passengers,
            "pickup_datetime": datetime.combine(trip_date, trip_time).isoformat(),
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        except requests.RequestException:
            st.error(f"Couldn't reach the API at {API_URL}. Is it running?")
        else:
            if response.status_code == 200:
                result = response.json()
                c1, c2 = st.columns(2)
                c1.metric("Predicted fare", f"${result['predicted_fare']:.2f}")
                km = result["trip_distance_km"]
                c2.metric("Straight-line distance", f"{km:.1f} km ({km * 0.621:.1f} miles)")
                if result.get("note"):
                    st.info(result["note"])
                st.caption("Typical error is about ±$1.45. Distance is measured as the crow flies; "
                           "the real route is longer.")
            else:
                st.error(error_message(response))

st.divider()
st.caption("Built with scikit-learn, FastAPI and Streamlit · Data: NYC taxi trips 2009–2015 · "
           "Fares are in 2009–2015 dollars, so today's fares would be higher.")
