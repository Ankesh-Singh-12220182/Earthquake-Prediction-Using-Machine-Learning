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
from tqdm import tqdm
from collections import Counter

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

# --- STREAMLIT APP ---
st.set_page_config(page_title="Earthquake Prediction App", layout="centered")
st.title("🌍 Earthquake Prediction Using Machine Learning")

# Magnitude Category Info
st.markdown("### 📊 Earthquake Category vs Magnitude Range")
category_magnitude_range = {
    'Micro': '< 2.0',
    'Minor': '2.0 - 4.0',
    'Light': '4.0 - 5.0',
    'Moderate': '5.0 - 6.0',
    'Strong': '6.0 - 7.0',
    'Major': '7.0 - 8.0',
    'Great': '>= 8.0'
}
st.write(category_magnitude_range)

# Load Data
data_load_state = st.text("Scraping earthquake data...")
data = scrape_earthquake_data()
processed_data = preprocess_data(data)
model, accuracy, report = train_model(processed_data)
data_load_state.text("Data loaded and model trained successfully!")

# Show accuracy
st.success(f"Model Trained! Accuracy on test data: {accuracy:.2f}")

# User Input Section
st.markdown("### 📥 Enter Details for Earthquake Prediction")
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0)
depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0)
date = st.date_input("Date")
time = st.time_input("Time")

# Feature Extraction
year = date.year
month = date.month
day = date.day
hour = time.hour
minute = time.minute
second = time.second

# Prediction
if st.button("Predict Earthquake Category"):
    input_df = pd.DataFrame({
        'Latitude': [latitude],
        'Longitude': [longitude],
        'Depth': [depth],
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Minute': [minute],
        'Second': [second]
    })

    probabilities = model.predict_proba(input_df)[0]
    class_labels = model.classes_
    top_idx = np.argmax(probabilities)
    predicted_class = class_labels[top_idx]
    confidence = probabilities[top_idx]

    st.info(f"🧭 Predicted Category: **{predicted_class}**\n🔒 Confidence: **{confidence:.2f}**")

    with st.expander("🔬 All Class Probabilities"):
        st.write({label: f"{prob:.2f}" for label, prob in zip(class_labels, probabilities)})

# --- AUTOMATIC PREDICTIONS FOR NEXT 7 DAYS ---
st.markdown("### 🔮 Predicted Earthquakes in Next 7 Days (Moderate & Above)")

from tqdm import tqdm

# Generate next 7 days
future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]

# Define expanded ranges
depths = [depth + d for d in range(-30, 31, 5) if 0 <= depth + d <= 700]
lat_range = [round(latitude + d, 2) for d in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3] if -90 <= latitude + d <= 90]
lon_range = [round(longitude + d, 2) for d in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3] if -180 <= longitude + d <= 180]
time_intervals = range(0, 24, 18)  # Every 4 hours

predictions = []
total_steps = len(future_dates) * len(time_intervals) * len(depths) * len(lat_range) * len(lon_range)
progress = st.progress(0)
step = 0

for future_date in future_dates:
    for hour in time_intervals:
        for d in depths:
            for lat in lat_range:
                for lon in lon_range:
                    input_df = pd.DataFrame({
                        'Latitude': [lat],
                        'Longitude': [lon],
                        'Depth': [d],
                        'Year': [future_date.year],
                        'Month': [future_date.month],
                        'Day': [future_date.day],
                        'Hour': [hour],
                        'Minute': [0],
                        'Second': [0]
                    })

                    probabilities = model.predict_proba(input_df)[0]
                    class_labels = model.classes_
                    top_idx = np.argmax(probabilities)
                    predicted_class = class_labels[top_idx]
                    confidence = probabilities[top_idx]

                    if predicted_class in ['Moderate', 'Strong', 'Major', 'Great']:
                        predictions.append({
                            'Date': future_date.date(),
                            'Time': f"{hour:02d}:00",
                            'Latitude': lat,
                            'Longitude': lon,
                            'Depth': d,
                            'Predicted Category': predicted_class,
                            'Confidence': round(confidence, 2),
                            'Class Probabilities': {label: f"{prob:.2f}" for label, prob in zip(class_labels, probabilities)}
                        })
                    step += 1
                    progress.progress(min(step / total_steps, 1.0))

# Filter predictions within 1000 km
nearby_predictions = []
for pred in predictions:
    distance = geodesic((latitude, longitude), (pred['Latitude'], pred['Longitude'])).km
    if distance <= 1000:
        pred['Distance (km)'] = round(distance, 2)
        nearby_predictions.append(pred)

# Display results
st.write(f"Predicted earthquakes (Moderate or above) with user input automation within 1000 km radius in next 7 days:")
if nearby_predictions:
    st.dataframe(pd.DataFrame(nearby_predictions).reset_index(drop=True), use_container_width=True)

else:
    st.info("✅ No significant earthquakes predicted in your area over the next 7 days.")


# Optional: Show raw data
with st.expander("🔍 Preview Recent Earthquake Data"):
    st.dataframe(processed_data[['Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude', 'Location']].head(20))


# --- MAP OF RECENT EARTHQUAKES ---
st.markdown("### 🗺️ Recent Earthquake Locations on World Map")

# Rename columns for compatibility with st.map()
map_data = processed_data[['Latitude', 'Longitude']].copy()
map_data.columns = ['latitude', 'longitude']

# Display simple world map
st.map(map_data)

# Optional: Enhanced Interactive Map with pydeck
import pydeck as pdk

# Prepare pydeck map data with magnitudes as circle radius
map_df = processed_data[['Latitude', 'Longitude', 'Magnitude']].copy()
map_df.columns = ['lat', 'lon', 'mag']

layer = pdk.Layer(
    'ScatterplotLayer',
    data=map_df,
    get_position='[lon, lat]',
    get_radius='mag * 10000',
    get_fill_color='[255, 0, 0, 140]',
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=map_df['lat'].mean(),
    longitude=map_df['lon'].mean(),
    zoom=1,
    pitch=0,
)

st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9', initial_view_state=view_state, layers=[layer]))