import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import random
import time

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="EcoHealth System", layout="wide")

# -----------------------------
# PREMIUM CSS
# -----------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1,h2,h3,h4,p,label { color: white !important; }
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    transition: 0.3s;
}
.card:hover { transform: scale(1.03); }
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white; border-radius:10px;
}
[data-testid="stDataFrame"] { background:white; color:black; }
.js-plotly-plot text { fill:white !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# REQUIRED FEATURES (7)
# -----------------------------
REQUIRED = [
    "Temperature","Humidity","AirQuality",
    "Soil_Moisture","Light","pH","Hour"
]

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(df):
    df.columns = df.columns.str.strip()

    df.rename(columns={
        "Soil Moisture":"Soil_Moisture",
        "Air Quality":"AirQuality",
        "PH":"pH"
    }, inplace=True)

    if "Timestamp" in df.columns:
        df["Hour"] = pd.to_datetime(df["Timestamp"],errors='coerce').dt.hour
    else:
        df["Hour"] = 0

    return df.fillna(0)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return (
            joblib.load("ecosystem_rf_model.pkl"),
            joblib.load("scaler.pkl"),
            joblib.load("label_encoder.pkl")
        )
    except:
        return None,None,None

model, scaler, encoder = load_model()

# -----------------------------
# LOGIN
# -----------------------------
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u=="admin" and p=="admin123":
            st.session_state.login=True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🌿 EcoHealth")
page = st.sidebar.radio("Navigation",
    ["Home","Dashboard","Prediction","Analytics","IoT","Report","About"])

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# -----------------------------
# HOME
# -----------------------------
if page == "Home":

    # 🌍 TITLE
    st.markdown("<h1 style='text-align:center;'>🌍 EcoHealth Monitoring System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>AI + IoT Based Ecosystem Analysis</h4>", unsafe_allow_html=True)

    st.markdown("---")

    # 🌱 INTRODUCTION
    st.markdown("""
    ### 🌱 Introduction

    An ecosystem is a community of living organisms interacting with their physical environment.
    Monitoring ecosystem health is essential for maintaining biodiversity, environmental balance,
    and sustainability.

    This system uses *Machine Learning and IoT simulation* to analyze environmental parameters
    and predict ecosystem health.
    """)

    st.markdown("---")

    # 🌵 DESERT ECOSYSTEM
    col1, col2 = st.columns(2)

    with col1:
        st.image("https://images.unsplash.com/photo-1501785888041-af3ef285b470")
    
    with col2:
        st.markdown("""
        <div class='card'>
        <h3>🌵 Desert Ecosystem</h3>

        <b>Definition:</b>  
        A desert ecosystem is characterized by extremely low rainfall, high temperature variation,
        and limited vegetation.

        <b>Features:</b>
        - Very low precipitation  
        - Extreme heat during day, cold nights  
        - Sparse vegetation (cactus, shrubs)  

        <b>Importance:</b>
        - Supports unique biodiversity  
        - Maintains ecological balance  
        - Helps in climate regulation  

        <b>Challenges:</b>
        - Water scarcity  
        - Desertification  
        - Climate change impact  
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 🌊 AQUATIC ECOSYSTEM
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='card'>
        <h3>🌊 Aquatic Ecosystem</h3>

        <b>Definition:</b>  
        Aquatic ecosystems are water-based environments including oceans, rivers, lakes, and wetlands.

        <b>Types:</b>
        - Freshwater ecosystem  
        - Marine ecosystem  

        <b>Features:</b>
        - Water as main medium  
        - High biodiversity  
        - Regulates temperature  

        <b>Importance:</b>
        - Provides oxygen (phytoplankton)  
        - Supports marine life  
        - Maintains global climate  

        <b>Challenges:</b>
        - Water pollution  
        - Overfishing  
        - Global warming  
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e")

    st.markdown("---")

    # 🧠 SYSTEM FEATURES
    st.markdown("""
    ### 🧠 System Features

    - 🤖 Machine Learning based prediction (Random Forest)
    - 📊 Data Analytics and Visualization
    - 📡 IoT Sensor Simulation
    - 📄 Automated Report Generation

    ### 🎯 Objective

    To develop an intelligent system that monitors environmental parameters
    and predicts ecosystem health for better decision-making.
    """)

# -----------------------------
# DASHBOARD
# -----------------------------
elif page=="Dashboard":
    if uploaded:
        df = preprocess(pd.read_csv(uploaded))

        c1,c2,c3 = st.columns(3)
        c1.markdown(f"<div class='card'><h3>Temp</h3><h2>{df['Temperature'].mean():.2f}</h2></div>",unsafe_allow_html=True)
        c2.markdown(f"<div class='card'><h3>Humidity</h3><h2>{df['Humidity'].mean():.2f}</h2></div>",unsafe_allow_html=True)
        c3.markdown(f"<div class='card'><h3>AQI</h3><h2>{df['AirQuality'].mean():.2f}</h2></div>",unsafe_allow_html=True)

        st.dataframe(df.head())

# -----------------------------
# PREDICTION
# -----------------------------
elif page=="Prediction":
    if uploaded:
        df = preprocess(pd.read_csv(uploaded))
        X = df[REQUIRED]

        preds = model.predict(scaler.transform(X))
        df["Prediction"] = encoder.inverse_transform(preds)

        st.success("Prediction done")
        st.dataframe(df)

# -----------------------------
# ANALYTICS (FULL)
# -----------------------------
elif page=="Analytics":
    if uploaded:
        df = preprocess(pd.read_csv(uploaded))

        st.write(df.describe())

        # Heatmap
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm",ax=ax)
        st.pyplot(fig)

        # Scatter
        fig = px.scatter(df,x="Temperature",y="Humidity",
                         color="AirQuality",template="plotly_dark")
        st.plotly_chart(fig)

        # Histogram
        fig = px.histogram(df,x="Temperature",template="plotly_dark")
        st.plotly_chart(fig)

        # Feature importance
        if model:
            imp = model.feature_importances_
            features = REQUIRED if len(imp)==len(REQUIRED) else [f"F{i}" for i in range(len(imp))]

            fig = px.bar(x=features,y=imp,template="plotly_dark")
            st.plotly_chart(fig)

# -----------------------------
# IoT
# -----------------------------
elif page=="IoT":
    chart = st.empty()
    data=[]
    for i in range(15):
        data.append(random.uniform(20,40))
        fig = px.line(y=data,template="plotly_dark")
        chart.plotly_chart(fig)
        time.sleep(0.3)

# -----------------------------
# REPORT
# -----------------------------
elif page=="Report":
    if uploaded:
        df = preprocess(pd.read_csv(uploaded))
        X = df[REQUIRED]

        preds = model.predict(scaler.transform(X))
        df["Prediction"] = encoder.inverse_transform(preds)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200,10,"Ecosystem Report",ln=True)
        pdf.cell(200,10,f"Records: {len(df)}",ln=True)

        for k,v in df["Prediction"].value_counts().items():
            pdf.cell(200,10,f"{k}: {v}",ln=True)

        pdf_bytes = pdf.output(dest='S').encode('latin1')
        st.download_button("Download Report", pdf_bytes, "report.pdf")

# -----------------------------
# ABOUT
# -----------------------------
elif page=="About":
    st.markdown("""
    ### Ecosystem Health Prediction System
    
    - Machine Learning Model  
    - Data Analytics  
    - IoT Simulation  
    - PDF Reporting  
    """)