import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import random
import time

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="EcoHealth Premium", layout="wide")

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
# REQUIRED FEATURES
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
# SAFE FEATURE HANDLER
# -----------------------------
def get_features(df):
    for col in REQUIRED:
        if col not in df.columns:
            df[col] = 0
    return df[REQUIRED]

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
# SAMPLE DATASET DOWNLOAD
# -----------------------------
def generate_sample_data():
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "Temperature": np.random.uniform(20, 40, 100).round(2),
        "Humidity": np.random.uniform(30, 90, 100).round(2),
        "AirQuality": np.random.randint(50, 200, 100),
        "Soil_Moisture": np.random.uniform(10, 60, 100).round(2),
        "Light": np.random.uniform(100, 1000, 100).round(2),
        "pH": np.random.uniform(5, 8, 100).round(2),
        "Timestamp": pd.date_range(start="2024-01-01", periods=100, freq="H")
    })

    return df.to_csv(index=False).encode("utf-8")

sample_csv = generate_sample_data()

st.sidebar.download_button(
    label="⬇ Download Sample Dataset",
    data=sample_csv,
    file_name="sample_ecosystem_data.csv",
    mime="text/csv"
)
# -----------------------------
# LOGOUT BUTTON
# -----------------------------
if st.sidebar.button("🚪 Logout"):
    st.session_state.login = False
    st.rerun()

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
elif page == "Dashboard":

    st.title("📊 Dashboard Overview")

    if uploaded is None:
        st.warning("Please upload dataset")
    else:
        df = pd.read_csv(uploaded)
        df = preprocess(df)

        # -----------------------------
        # METRICS CARDS
        # -----------------------------
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"<div class='card'><h3>🌡 Temperature</h3><h2>{df['Temperature'].mean():.2f}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'><h3>💧 Humidity</h3><h2>{df['Humidity'].mean():.2f}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'><h3>🌫 Air Quality</h3><h2>{df['AirQuality'].mean():.2f}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")

        # -----------------------------
        # PREDICTION FOR GAUGE
        # -----------------------------
        if model:

            # Safe features
            for col in REQUIRED:
                if col not in df.columns:
                    df[col] = 0

            X = df[REQUIRED]

            preds = model.predict(scaler.transform(X))
            df["Prediction"] = encoder.inverse_transform(preds)

# -----------------------------
# HEALTH SCORE MAPPING
# -----------------------------
mapping = {
    "Healthy": 90,
    "Moderate": 60,
    "Critical": 30
}

df["Score"] = df["Prediction"].map(mapping).fillna(50)
avg_score = df["Score"].mean()

# -----------------------------
# STATUS TEXT
# -----------------------------
if avg_score >= 75:
    status = "🟢 Healthy"
elif avg_score >= 50:
    status = "🟡 Moderate"
else:
    status = "🔴 At Risk"

st.subheader(f"🧭 Ecosystem Status: {status}")

# -----------------------------
# GAUGE METER
# -----------------------------
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=avg_score,
    number={'suffix': " /100"},
    title={'text': "Ecosystem Health Score"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {
            'color': "green" if avg_score >= 75 else ("orange" if avg_score >= 50 else "red")
        },
        'steps': [
            {'range': [0, 50], 'color': "lightcoral"},   # At Risk
            {'range': [50, 75], 'color': "gold"},        # Moderate
            {'range': [75, 100], 'color': "lightgreen"}  # Healthy
        ],
    }
))

st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # DATA PREVIEW
        # -----------------------------
        st.subheader("📋 Data Preview")
        st.dataframe(df.head())



# -----------------------------
# PREDICTION
# -----------------------------
elif page == "Prediction":

    st.title("🔮 Ecosystem Health Prediction")

    st.markdown("""
    ### 📘 About Prediction

    This module uses a *Machine Learning model (Random Forest)* to predict 
    the health of an ecosystem based on environmental parameters.

    ### ⚙️ Features Used:
    - Temperature  
    - Humidity  
    - Air Quality  
    - Soil Moisture  
    - Light Intensity  
    - pH Level  
    - Hour (time-based feature)  

    ### 🎯 Output Classes:
    - 🟢 Healthy → Stable environment  
    - 🟡 Moderate → Slight imbalance  
    - 🔴 Critical (At Risk) → Environmental stress  

    ---
    """)

    if uploaded is None:
        st.warning("Please upload dataset to perform prediction")
    else:
        df = pd.read_csv(uploaded)
        df = preprocess(df)

        try:
            # -----------------------------
            # SAFE FEATURE HANDLING
            # -----------------------------
            for col in REQUIRED:
                if col not in df.columns:
                    df[col] = 0

            X = df[REQUIRED]

            # -----------------------------
            # MODEL PREDICTION
            # -----------------------------
            preds = model.predict(scaler.transform(X))
            df["Prediction"] = encoder.inverse_transform(preds)

            # -----------------------------
            # CLEAN OUTPUT
            # -----------------------------
            result_df = df[REQUIRED + ["Prediction"]]

            st.success("✅ Prediction completed successfully")

            st.subheader("📋 Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            # -----------------------------
            # INTERPRETATION
            # -----------------------------
            st.markdown("### 🧠 Interpretation")

            counts = result_df["Prediction"].value_counts()

            for label, count in counts.items():
                st.write(f"• {label}: {count} records")

            st.markdown("""
            ✔ If most values are *Healthy*, ecosystem is stable  
            ✔ If many are *Moderate*, monitoring is required  
            ✔ If *Critical*, immediate action is needed  
            """)

            # -----------------------------
            # OPTIONAL DOWNLOAD
            # -----------------------------
            st.download_button(
                "⬇ Download Prediction Results",
                result_df.to_csv(index=False),
                "prediction_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------
# ANALYTICS
# -----------------------------
elif page == "Analytics":

    st.title("📈 Advanced Analytics Dashboard")

    if uploaded is None:
        st.warning("Please upload dataset to view analytics")
    else:
        df = preprocess(pd.read_csv(uploaded))

        # -----------------------------
        # 📊 SUMMARY
        # -----------------------------
        st.markdown("## 📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown("---")
# -----------------------------
# 🔥 INTERACTIVE HEATMAP
# -----------------------------
st.markdown("## 🔥 Correlation Heatmap (Custom Selection)")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

selected_cols = st.multiselect(
    "Select features for correlation",
    numeric_cols,
    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
)

if len(selected_cols) >= 2:

    corr_data = df[selected_cols].corr()

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        corr_data,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Selected Feature Correlation", fontsize=14, color="white")

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    fig.patch.set_facecolor('#0f2027')
    ax.set_facecolor('#0f2027')

    st.pyplot(fig)

else:
    st.warning("Please select at least 2 features")

        
        # -----------------------------
        # 🔍 SCATTER (FIXED)
        # -----------------------------
        st.markdown("## 🔍 Feature Relationship Analysis")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        x_axis = st.selectbox("Select X-axis", numeric_cols)
        y_axis = st.selectbox("Select Y-axis", numeric_cols, index=1)

        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color="AirQuality" if "AirQuality" in df.columns else None,
            title=f"{x_axis} vs {y_axis}",
            template="plotly_dark"
        )

        fig.update_layout(
            title_x=0.5,
            xaxis=dict(
                title=x_axis,
                showgrid=True,
                gridcolor='gray',
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                title=y_axis,
                showgrid=True,
                gridcolor='gray',
                tickfont=dict(color='white')
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # -----------------------------
        # 📉 TREND (FIXED)
        # -----------------------------
        st.markdown("## 📉 Trend Analysis")

        trend_feature = st.selectbox("Select Feature", numeric_cols, key="trend")

        fig = px.line(
            df,
            y=trend_feature,
            title=f"{trend_feature} Trend",
            template="plotly_dark"
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title="Index",
            yaxis_title=trend_feature,
            xaxis=dict(showgrid=True, gridcolor='gray'),
            yaxis=dict(showgrid=True, gridcolor='gray')
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # -----------------------------
        # 📊 HISTOGRAM (FIXED)
        # -----------------------------
        st.markdown("## 📊 Feature Distribution")

        hist_feature = st.selectbox("Select Feature", numeric_cols, key="hist")

        fig = px.histogram(
            df,
            x=hist_feature,
            nbins=20,
            title=f"{hist_feature} Distribution",
            template="plotly_dark"
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title=hist_feature,
            yaxis_title="Count",
            xaxis=dict(showgrid=True, gridcolor='gray'),
            yaxis=dict(showgrid=True, gridcolor='gray')
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # -----------------------------
        # 🧠 FEATURE IMPORTANCE
        # -----------------------------
        if model:
            st.markdown("## 🧠 Feature Importance")

            try:
                imp = model.feature_importances_

                if len(imp) == len(REQUIRED):
                    features = REQUIRED
                else:
                    features = [f"Feature {i}" for i in range(len(imp))]

                imp_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": imp
                })

                fig = px.bar(
                    imp_df,
                    x="Feature",
                    y="Importance",
                    title="Model Feature Importance",
                    template="plotly_dark"
                )

                fig.update_layout(
                    title_x=0.5,
                    xaxis_title="Features",
                    yaxis_title="Importance Score",
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True)
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Feature importance not available: {e}")

        st.markdown("---")

        # -----------------------------
        # 🚨 OUTLIER DETECTION
        # -----------------------------
        st.markdown("## 🚨 Outlier Detection")

        out_feature = st.selectbox("Select Feature", numeric_cols, key="outlier")

        Q1 = df[out_feature].quantile(0.25)
        Q3 = df[out_feature].quantile(0.75)
        IQR = Q3 - Q1

        outliers = df[
            (df[out_feature] < Q1 - 1.5 * IQR) |
            (df[out_feature] > Q3 + 1.5 * IQR)
        ]

        st.write(f"🔴 Total Outliers in {out_feature}: {len(outliers)}")

        st.markdown("---")

        # -----------------------------
        # 💡 INSIGHTS
        # -----------------------------
        st.markdown("## 💡 Smart Insights")

        insights = []

        if "Temperature" in df.columns and df["Temperature"].mean() > 35:
            insights.append("🌡 High temperature → ecosystem stress")

        if "Humidity" in df.columns and df["Humidity"].mean() < 40:
            insights.append("💧 Low humidity → dry conditions")

        if "AirQuality" in df.columns and df["AirQuality"].mean() > 150:
            insights.append("🌫 Poor air quality → critical condition")

        if not insights:
            insights.append("✅ Environment appears stable")

        for i in insights:
            st.success(i)

# -----------------------------
# IoT
# -----------------------------
elif page == "IoT":

    st.title("📡 IoT Simulation")

    st.markdown("Simulating real-time sensor data...")

    # Create placeholders
    metric1, metric2, metric3 = st.columns(3)
    chart = st.empty()

    data = []

    for i in range(20):

        # Generate simple random data
        temp = random.uniform(20, 40)
        hum = random.uniform(30, 90)
        aqi = random.randint(50, 200)

        # Show metrics
        metric1.metric("🌡 Temperature", f"{temp:.2f} °C")
        metric2.metric("💧 Humidity", f"{hum:.2f} %")
        metric3.metric("🌫 AQI", f"{aqi}")

        # Update chart (only temperature for simplicity)
        data.append(temp)

        fig = px.line(y=data, title="Temperature Trend", template="plotly_dark")
        chart.plotly_chart(fig, use_container_width=True)

        time.sleep(0.5)

# -----------------------------
# REPORT
# -----------------------------
elif page == "Report":

    st.title("📄 Generate Report")

    if uploaded is None:
        st.warning("Please upload dataset first")
    else:
        df = pd.read_csv(uploaded)
        df = preprocess(df)

        try:
            # -----------------------------
            # SAFE FEATURES
            # -----------------------------
            for col in REQUIRED:
                if col not in df.columns:
                    df[col] = 0

            X = df[REQUIRED]

            # -----------------------------
            # PREDICTION
            # -----------------------------
            preds = model.predict(scaler.transform(X))
            df["Prediction"] = encoder.inverse_transform(preds)

            # -----------------------------
            # CREATE GRAPH (Temperature Trend)
            # -----------------------------
            fig = plt.figure()
            plt.plot(df["Temperature"], color="blue")
            plt.title("Temperature Trend")
            plt.xlabel("Index")
            plt.ylabel("Temperature")

            graph_path = "temp_plot.png"
            plt.savefig(graph_path)
            plt.close()

            # -----------------------------
            # PDF
            # -----------------------------
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, "Ecosystem Health Report", ln=True)
            pdf.cell(200, 10, f"Total Records: {len(df)}", ln=True)

            pdf.cell(200, 10, f"Avg Temperature: {df['Temperature'].mean():.2f}", ln=True)
            pdf.cell(200, 10, f"Avg Humidity: {df['Humidity'].mean():.2f}", ln=True)
            pdf.cell(200, 10, f"Avg Air Quality: {df['AirQuality'].mean():.2f}", ln=True)

            pdf.cell(200, 10, "Prediction Summary:", ln=True)

            for k, v in df["Prediction"].value_counts().items():
                pdf.cell(200, 10, f"{k}: {v}", ln=True)

            # -----------------------------
            # ADD GRAPH TO PDF
            # -----------------------------
            pdf.ln(5)
            pdf.cell(200, 10, "Temperature Trend Graph:", ln=True)
            pdf.image(graph_path, x=10, w=180)

            # -----------------------------
            # DOWNLOAD
            # -----------------------------
            pdf_bytes = pdf.output(dest='S').encode('latin1')

            st.download_button(
                "⬇ Download Report",
                pdf_bytes,
                "ecosystem_report.pdf",
                mime="application/pdf"
            )

            st.success("✅ Report generated!")

        except Exception as e:
            st.error(f"Error: {e}")
# -----------------------------
# ABOUT
# -----------------------------
elif page == "About":

    st.title("About the Project")

    st.markdown("""
    ### 🌍 EcoHealth Prediction System

    This project is a *Machine Learning-based web application* designed to monitor 
    and analyze environmental conditions using various ecosystem parameters such as 
    temperature, humidity, air quality, soil moisture, light intensity, and pH level.

    The system predicts the *health status of an ecosystem* and provides meaningful 
    insights through data visualization and reports.

    ---
    
    ### 🎯 Objectives

    - To analyze environmental data using Machine Learning  
    - To predict ecosystem health status  
    - To visualize data using interactive dashboards  
    - To simulate IoT-based real-time monitoring  
    - To generate automated reports  

    ---
    
    ### ⚙️ Technologies Used

    - Python  
    - Streamlit  
    - Scikit-learn (Machine Learning)  
    - Pandas & NumPy  
    - Plotly & Matplotlib  
    - FPDF (PDF Generation)  

    ---
    
    ### 🔍 Key Features

    - 📊 Interactive Dashboard  
    - 🔮 Ecosystem Health Prediction  
    - 📈 Advanced Analytics & Visualization  
    - 📡 IoT Sensor Simulation  
    - 📄 Automated PDF Report Generation  

    ---
    
    ### 🚀 Future Scope

    - Integration with real IoT sensors (Arduino, Raspberry Pi)  
    - Cloud database integration  
    - Mobile application development  
    - AI-based alert system for critical conditions  
    - Real-time monitoring system  

    ---
    
    ### 👩‍💻 Developed By

    *Priyal Choudhary*  
    BCA Final Year Student  

    ---
    
    ### 📌 Conclusion

    This system demonstrates the practical implementation of Machine Learning, 
    Data Analytics, and IoT concepts to solve real-world environmental monitoring problems.
    """)