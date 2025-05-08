import streamlit as st
import pandas as pd
import numpy as np
import requests
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime, timedelta
from geopy.distance import geodesic
from collections import Counter

# --- STYLING ---
st.set_page_config(page_title="Earthquake Prediction App", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
            color: #2c3e50;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton > button {
            background-color: #1abc9c;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1em;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #16a085;
            transform: scale(1.05);
        }
        .stTextInput > div > input,
        .stNumberInput > div > input,
        .stDateInput > div > input,
        .stTimeInput > div > input {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 0.4em;
        }
        .stExpanderHeader:hover {
            color: #1abc9c;
            font-weight: bold;
        }
        .stMarkdown h3, .stMarkdown h2, .stMarkdown h1 {
            color: #1abc9c;
        }
    </style>
""", unsafe_allow_html=True)

# --- TITLE WITH EMOJI ---
st.markdown("<h1 style='text-align: center;'>🌍 Earthquake Prediction App 🚨</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>🔍 Using Machine Learning to Forecast Seismic Events</h5>", unsafe_allow_html=True)
st.markdown("---")

# --- CATEGORY FUNCTION ---
def classify_magnitude(mag):
    if mag < 2.0:
        return 'Micro'
    elif 2.0 <= mag < 4.0:
        return 'Minor'
    elif 4.0 <= mag < 5.0:
        return 'Light'
    elif 5.0 <= mag < 6.0:
        return 'Moderate'
    elif 6.0 <= mag < 7.0:
        return 'Strong'
    elif 7.0 <= mag < 8.0:
        return 'Major'
    else:
        return 'Great'

# --- SCRAPE DATA ---
@st.cache_data
def scrape_earthquake_data():
    url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv'
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(url)
    else:
        st.error(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

# --- PREPROCESS DATA ---
def preprocess_data(df):
    df = df[['time', 'latitude', 'longitude', 'depth', 'mag', 'place']].copy()
    df.columns = ['Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude', 'Location']
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df.dropna(inplace=True)
    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month
    df['Day'] = df['Time'].dt.day
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    df['Second'] = df['Time'].dt.second
    df['Magnitude_Class'] = df['Magnitude'].apply(classify_magnitude)
    return df

# --- TRAIN MODEL ---
@st.cache_resource
def train_model(df):
    features = ['Latitude', 'Longitude', 'Depth', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']
    X = df[features]
    y = df['Magnitude_Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    class_counts = Counter(y_train)
    min_class_count = min(class_counts.values())
    if min_class_count > 1:
        smote_k = min(5, min_class_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=smote_k)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, accuracy, report

# --- LOAD DATA & TRAIN ---
with st.spinner("🔄 Loading earthquake data and training model..."):
    data = scrape_earthquake_data()
    processed_data = preprocess_data(data)
    model, accuracy, report = train_model(processed_data)
st.success(f"✅ Model trained successfully with accuracy on test data: {accuracy:.2f}")

# --- EARTHQUAKE CATEGORY TABLE ---
st.markdown("### 📊 Earthquake Category vs Magnitude Range")
st.write({
    'Micro': '< 2.0',
    'Minor': '2.0 - 4.0',
    'Light': '4.0 - 5.0',
    'Moderate': '5.0 - 6.0',
    'Strong': '6.0 - 7.0',
    'Major': '7.0 - 8.0',
    'Great': '>= 8.0'
})
st.markdown("---")

# --- USER INPUT ---
st.markdown("### 📝 Enter Details For Earthquake Prediction")
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0)
depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0)
date = st.date_input("Date")
time = st.time_input("Time")
year, month, day = date.year, date.month, date.day
hour, minute, second = time.hour, time.minute, time.second

if st.button("🚀 Predict Earthquake Category"):
    input_df = pd.DataFrame({
        'Latitude': [latitude], 'Longitude': [longitude], 'Depth': [depth],
        'Year': [year], 'Month': [month], 'Day': [day],
        'Hour': [hour], 'Minute': [minute], 'Second': [second]
    })
    probabilities = model.predict_proba(input_df)[0]
    class_labels = model.classes_
    predicted_class = class_labels[np.argmax(probabilities)]
    confidence = np.max(probabilities)

    st.toast("Hooray! Earthquake prediction completed 🎉", icon='🎈')
    st.info(f"🧠 **Predicted Category:** {predicted_class}\n🔐 **Confidence:** {confidence:.2f}")

    with st.expander("🔬 View All Class Probabilities"):
        st.write({label: f"{prob:.2f}" for label, prob in zip(class_labels, probabilities)})

# --- FUTURE PREDICTIONS ---
st.markdown("### 🔮 Predicted Earthquakes Near Your Location in Next 7 Days (Magnitude ≥ 5 , Confidence ≥ 0.5)")
future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
depths = [depth + d for d in range(-30, 31, 5) if 0 <= depth + d <= 700]
lat_range = [round(latitude + d, 2) for d in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3] if -90 <= latitude + d <= 90]
lon_range = [round(longitude + d, 2) for d in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3] if -180 <= longitude + d <= 180]
time_intervals = range(0, 24, 6)

predictions = []
progress = st.progress(0)
step, total_steps = 0, len(future_dates) * len(time_intervals) * len(depths) * len(lat_range) * len(lon_range)

for future_date in future_dates:
    for hour in time_intervals:
        for d in depths:
            for lat in lat_range:
                for lon in lon_range:
                    input_df = pd.DataFrame({
                        'Latitude': [lat], 'Longitude': [lon], 'Depth': [d],
                        'Year': [future_date.year], 'Month': [future_date.month], 'Day': [future_date.day],
                        'Hour': [hour], 'Minute': [0], 'Second': [0]
                    })
                    probs = model.predict_proba(input_df)[0]
                    pred_class = model.classes_[np.argmax(probs)]
                    confidence = np.max(probs)
                    if pred_class in ['Moderate', 'Strong', 'Major', 'Great'] and confidence >= 0.5:
                        predictions.append({
                            'Date': future_date.date(), 'Time': f"{hour:02d}:00",
                            'Latitude': lat, 'Longitude': lon, 'Depth': d,
                            'Predicted Category': pred_class, 'Confidence': round(confidence, 2),
                            'Class Probabilities': {label: f"{prob:.2f}" for label, prob in zip(model.classes_, probs)}
                        })
                    step += 1
                    progress.progress(min(step / total_steps, 1.0))

nearby = []
for pred in predictions:
    dist = geodesic((latitude, longitude), (pred['Latitude'], pred['Longitude'])).km
    if dist <= 1000:
        pred['Distance (km)'] = round(dist, 2)
        nearby.append(pred)

if nearby:
    st.dataframe(pd.DataFrame(nearby).reset_index(drop=True), use_container_width=True)
else:
    st.info("✅ No significant earthquakes predicted in your area over the next 7 days.")

# --- DATA PREVIEW ---
with st.expander("🔍 Preview Recent Earthquake Data"):
    st.dataframe(processed_data[['Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude', 'Location']].head(20))

# --- MAPS ---
st.markdown("### 🗺️ Recent Earthquake Locations on Map")
map_data = processed_data[['Latitude', 'Longitude']].copy()
map_data.columns = ['latitude', 'longitude']
st.map(map_data)

# --- PYDECK MAP ---
map_df = processed_data[['Latitude', 'Longitude', 'Magnitude', 'Location']].copy()
map_df.columns = ['lat', 'lon', 'mag', 'location']

tooltip = {
    "html": "<b>Magnitude:</b> {mag} <br/><b>Location:</b> {location}",
    "style": {"color": "white"}
}

layer = pdk.Layer(
    'ScatterplotLayer',
    data=map_df,
    get_position='[lon, lat]',
    get_radius='mag * 10000',
    get_fill_color='[255, 0, 0, 140]',
    pickable=True
)

view_state = pdk.ViewState(latitude=map_df['lat'].mean(), longitude=map_df['lon'].mean(), zoom=1, pitch=0)

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=view_state,
    layers=[layer],
    tooltip=tooltip
))
