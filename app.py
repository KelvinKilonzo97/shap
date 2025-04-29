import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
import hashlib
from datetime import datetime
from meteostat import Point, Hourly
from tensorflow.keras.models import load_model
import holidays
import shap
import matplotlib.pyplot as plt

# =============================
# ✅ Streamlit Configuration
# =============================
st.set_page_config("Flight Delay Predictor", page_icon="✈️", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🛫 Smart Flight Delay Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Real-Time Delay Insights Using AI + Live Weather</h4>", unsafe_allow_html=True)

# =============================
# ✅ Load model and scaler
# =============================
@st.cache_resource
def load_resources():
    model = load_model("flight_delay_lstm_model2.h5", compile=False)
    scaler = joblib.load("lstm_scaler2.pkl")
    return model, scaler

model, scaler = load_resources()
us_holidays = holidays.US()

# =============================
# ✅ Airport & Carrier Mappings
# =============================
airport_data = {
    'ATL': (33.6407, -84.4277), 'CLT': (35.2140, -80.9431), 'DEN': (39.8561, -104.6737),
    'DFW': (32.8998, -97.0403), 'EWR': (40.6895, -74.1745), 'IAH': (29.9902, -95.3368),
    'JFK': (40.6413, -73.7781), 'LAS': (36.0840, -115.1537), 'LAX': (33.9416, -118.4085),
    'MCO': (28.4312, -81.3081), 'MIA': (25.7959, -80.2870), 'ORD': (41.9742, -87.9073),
    'PHX': (33.4373, -112.0078), 'SEA': (47.4502, -122.3088), 'SFO': (37.6213, -122.3790)
}
airport_mapping = {code: idx for idx, code in enumerate(airport_data.keys())}
airport_fullname_to_code = {
    "Hartsfield–Jackson Atlanta Intl (ATL)": "ATL", "Charlotte Douglas Intl (CLT)": "CLT",
    "Denver Intl (DEN)": "DEN", "Dallas/Fort Worth Intl (DFW)": "DFW", "Newark Liberty Intl (EWR)": "EWR",
    "George Bush Intercontinental (IAH)": "IAH", "John F. Kennedy Intl (JFK)": "JFK",
    "Harry Reid Intl (LAS)": "LAS", "Los Angeles Intl (LAX)": "LAX", "Orlando Intl (MCO)": "MCO",
    "Miami Intl (MIA)": "MIA", "O'Hare Intl (ORD)": "ORD", "Phoenix Sky Harbor Intl (PHX)": "PHX",
    "Seattle–Tacoma Intl (SEA)": "SEA", "San Francisco Intl (SFO)": "SFO"
}
carrier_fullname_to_code = {
    "American Airlines (AA)": "AA", "Alaska Airlines (AS)": "AS", "JetBlue Airways (B6)": "B6",
    "Delta Air Lines (DL)": "DL", "Frontier Airlines (F9)": "F9", "Allegiant Air (G4)": "G4",
    "Envoy Air (MQ)": "MQ", "Spirit Airlines (NK)": "NK", "PSA Airlines (OH)": "OH",
    "SkyWest Airlines (OO)": "OO", "United Airlines (UA)": "UA", "Southwest Airlines (WN)": "WN",
    "Republic Airways (YX)": "YX"
}
carrier_mapping = {code: idx for idx, code in enumerate(carrier_fullname_to_code.values())}

# =============================
# ✅ Weather & Delay Functions
# =============================
def get_weather_data(airport_code, date, time):
    try:
        lat, lon = airport_data[airport_code]
        location = Point(lat, lon)
        dt = datetime(date.year, date.month, date.day, time.hour)
        data = Hourly(location, dt, dt).fetch()
        if data.empty:
            st.warning("⚠️ No weather data available for the selected time/location.")
            return None
        row = data.iloc[0]
        return {
            "temperature": row['temp'], "humidity": row['rhum'], "wind_dir": row['wdir'],
            "wind_speed": row['wspd'], "pressure": row['pres']
        }
    except Exception as e:
        st.error(f"🌦️ Weather fetch error: {e}")
        return None

def get_synthetic_dep_delay(date, time, origin, dest, carrier):
    input_string = f"{date}_{time}_{origin}_{dest}_{carrier}"
    seed = int(hashlib.sha256(input_string.encode()).hexdigest(), 16) % (10**8)
    rng = random.Random(seed)
    buckets = [(-10, 0), (0, 15), (15, 45), (45, 90)]
    weights = [0.1, 0.5, 0.3, 0.1]
    selected_range = rng.choices(buckets, weights=weights, k=1)[0]
    return round(rng.uniform(*selected_range), 2)

# =============================
# ✅ Input Form
# =============================
st.markdown("### 📅 Flight Schedule")
col1, col2 = st.columns(2)
with col1: dep_date = st.date_input("Departure Date")
with col2: dep_time = st.time_input("Scheduled Departure Time")

col3, col4 = st.columns(2)
with col3: origin_full = st.selectbox("Origin Airport", list(airport_fullname_to_code.keys()))
with col4:
    valid_destinations = [a for a in airport_fullname_to_code.keys() if a != origin_full]
    dest_full = st.selectbox("Destination Airport", valid_destinations)

carrier_full = st.selectbox("Airline Carrier", list(carrier_fullname_to_code.keys()))

# =============================
# ✅ Feature Preparation
# =============================
def preprocess_input(dep_delay, weather):
    origin = airport_mapping[airport_fullname_to_code[origin_full]]
    dest = airport_mapping[airport_fullname_to_code[dest_full]]
    carrier = carrier_mapping[carrier_fullname_to_code[carrier_full]]
    date_obj = pd.to_datetime(dep_date)
    features = {
        "day": date_obj.day, "month": date_obj.month, "week": date_obj.isocalendar().week,
        "weekday": date_obj.weekday(), "holiday": int(dep_date in us_holidays), "hour": dep_time.hour
    }
    cyclical = {
        "month_sin": np.sin(2 * np.pi * features["month"] / 12),
        "month_cos": np.cos(2 * np.pi * features["month"] / 12),
        "weekday_sin": np.sin(2 * np.pi * features["weekday"] / 7),
        "weekday_cos": np.cos(2 * np.pi * features["weekday"] / 7),
        "dep_hour_sin": np.sin(2 * np.pi * features["hour"] / 24),
        "dep_hour_cos": np.cos(2 * np.pi * features["hour"] / 24)
    }
    return np.array([[ 
        origin, dest, dep_delay, carrier, weather["temperature"], weather["humidity"],
        weather["wind_dir"], weather["wind_speed"], weather["pressure"],
        features["day"], features["week"], features["month"], features["weekday"], features["holiday"],
        cyclical["month_sin"], cyclical["month_cos"], cyclical["weekday_sin"], cyclical["weekday_cos"],
        features["hour"], cyclical["dep_hour_sin"], cyclical["dep_hour_cos"]
    ]])

# =============================
# ✅ Prediction & SHAP
# =============================
prediction_output = None
last_input_vector = None

if st.button("🚀 Predict Arrival Delay"):
    origin_code = airport_fullname_to_code[origin_full]
    weather = get_weather_data(origin_code, dep_date, dep_time)
    if not weather:
        st.stop()
    dep_delay = get_synthetic_dep_delay(dep_date, dep_time, origin_full, dest_full, carrier_full)
    input_vector = preprocess_input(dep_delay, weather)
    last_input_vector = input_vector.copy()
    scaled = scaler.transform(input_vector)
    reshaped = scaled.reshape((1, 1, scaled.shape[1]))
    predicted_delay = model.predict(reshaped)[0][0]
    prediction_output = predicted_delay
    if predicted_delay < -1:
        st.success(f"🟢 Flight expected to arrive early by **{abs(predicted_delay):.2f} minutes**.")
    elif predicted_delay > 1:
        st.warning(f"🟠 Flight likely delayed by **{predicted_delay:.2f} minutes**.")
    else:
        st.info(f"🟡 Flight expected to be on time (±{predicted_delay:.2f} minutes).")

# =============================
# ✅ Explainable AI
# =============================
if last_input_vector is not None:
    if st.button("🔍 Explain This Prediction"):
        with st.spinner("Generating SHAP explanation..."):
            background = np.tile(last_input_vector, (100, 1))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(last_input_vector)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.initjs()
            st.markdown("#### SHAP Feature Contribution:")
            shap.force_plot(explainer.expected_value[0], shap_values[0][0], feature_names=[
                "origin", "dest", "dep_delay", "carrier", "temp", "humidity", "wind_dir", "wind_speed", "pressure",
                "day", "week", "month", "weekday", "holiday",
                "month_sin", "month_cos", "weekday_sin", "weekday_cos",
                "hour", "dep_hour_sin", "dep_hour_cos"
            ], matplotlib=True, show=True)
            st.pyplot(bbox_inches='tight')

# =============================
# ✅ Footer
# =============================
st.markdown("---")
st.markdown("<small><i>Created by Kelvin Ndambu Kilonzo | Strathmore University | v1.0</i></small>", unsafe_allow_html=True)
