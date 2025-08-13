import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# Load your trained pipeline
pipeline = joblib.load('pipeline.pkl')

# Page title
st.set_page_config(page_title="Forest Fire Area Prediction", page_icon="🔥", layout="wide")
st.title("🔥 Forest Fire Area Prediction Dashboard")

st.markdown("This tool predicts the **burned area** of forest fires based on various meteorological and environmental factors.")

# Sidebar for inputs
st.sidebar.header("Input Parameters")
x = st.sidebar.number_input("X coordinate", value=7)
y = st.sidebar.number_input("Y coordinate", value=5)
ffmc = st.sidebar.number_input("FFMC", value=86.2)
dmc = st.sidebar.number_input("DMC", value=26.2)
dc = st.sidebar.number_input("DC", value=94.3)
isi = st.sidebar.number_input("ISI", value=5.1)
temp = st.sidebar.number_input("Temperature", value=8.2)
rh = st.sidebar.number_input("Relative Humidity", value=51)
wind = st.sidebar.number_input("Wind", value=6.7)
rain = st.sidebar.number_input("Rain", value=0.0)
month = st.sidebar.number_input("Month (1-12)", min_value=1, max_value=12, value=9)
day = st.sidebar.number_input("Day (1-31)", min_value=1, max_value=31, value=15)

# Prepare DataFrame
row = pd.DataFrame([{
    'X': x,
    'Y': y,
    'FFMC': ffmc,
    'DMC': dmc,
    'DC': dc,
    'ISI': isi,
    'temp': temp,
    'RH': rh,
    'wind': wind,
    'rain': rain,
    'month': month,
    'day': day
}])
row['month'] = row['month'].astype(str)
row['day'] = row['day'].astype(str)

# Predict and display
if st.sidebar.button("Predict"):
    pred_log = pipeline.predict(row)[0]
    area_pred = np.exp(pred_log)

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Prediction Results")
        st.metric(label="Predicted Burned Area (ha)", value=f"{area_pred:.4f}")
        st.metric(label="Predicted log(area)", value=f"{pred_log:.4f}")

    with col2:
        # Gauge chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=area_pred,
            title={'text': "Burned Area (ha)"},
            gauge={'axis': {'range': [0, max(50, area_pred*1.5)]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

    # Bar chart comparison
    st.subheader("📈 Feature Values")
    st.bar_chart(row.drop(columns=['month', 'day']).T)

else:
    st.info("Enter values in the sidebar and click **Predict** to see results.")
