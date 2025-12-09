import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- CONFIG ---
st.set_page_config(page_title="Supply Chain Risk AI", page_icon="🚚", layout="wide")
RISK_THRESHOLD = 0.25

# --- 1. Train Data
@st.cache_resource
def train_model_live():
    # READ LOCAL FILE
    try:
        # This looks for the file in the same folder as app.py
        df = pd.read_csv('SCMS_Delivery_History_Dataset.csv', encoding='latin1')
    except FileNotFoundError:
        st.error("❌ Error: 'SCMS_Delivery_History_Dataset.csv' not found. Please upload it to GitHub!")
        return None, None, None, None, None

    # Cleanup
    if 'ID' in df.columns: df.rename(columns={'ID': 'id'}, inplace=True)
    elif 'ï»¿ID' in df.columns: df.rename(columns={'ï»¿ID': 'id'}, inplace=True)
    
    df['Scheduled Delivery Date'] = pd.to_datetime(df['Scheduled Delivery Date'], errors='coerce')
    df['Delivered to Client Date'] = pd.to_datetime(df['Delivered to Client Date'], errors='coerce')
    df = df.dropna(subset=['Scheduled Delivery Date', 'Delivered to Client Date'])
    
    df['Delay_Days'] = (df['Delivered to Client Date'] - df['Scheduled Delivery Date']).dt.days
    df['Is_Late'] = (df['Delay_Days'] > 0).astype(int)
    
    for col in ['Weight (Kilograms)', 'Freight Cost (USD)', 'Line Item Value', 'Line Item Quantity']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Shipment Mode'] = df['Shipment Mode'].fillna('Unknown')
    df['Scheduled_Month'] = df['Scheduled Delivery Date'].dt.month
    df['Scheduled_Year'] = df['Scheduled Delivery Date'].dt.year

    unique_vendors = df['Vendor'].unique()
    vendor_map = {name: f'Vendor_{i+1}' for i, name in enumerate(unique_vendors)}
    df['Vendor_Anonymized'] = df['Vendor'].map(vendor_map)

    # Features & Training
    feature_cols = ['Country', 'Shipment Mode', 'Vendor_Anonymized', 
                    'Line Item Quantity', 'Line Item Value', 'Weight (Kilograms)', 
                    'Freight Cost (USD)', 'Scheduled_Month', 'Scheduled_Year']
    X = df[feature_cols].copy()

    label_encoders = {}
    for col in ['Country', 'Shipment Mode', 'Vendor_Anonymized']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # "Paranoid Mode" (Class Weights)
    clf = RandomForestClassifier(n_estimators=60, max_depth=12, random_state=42, class_weight='balanced')
    clf.fit(X, df['Is_Late'])
    
    reg = RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42)
    reg.fit(X, df['Delay_Days'])
    
    return clf, reg, label_encoders, df, feature_cols

# Load the Brain
with st.spinner("🧠 Training AI Model on your data..."):
    clf, reg, encoders, history_df, feature_cols = train_model_live()

if clf is None:
    st.stop()

# --- 2. THE DASHBOARD UI ---
st.title("🚚 Unified Supply Chain Risk Predictor")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["📊 Executive View", "🔮 Risk Predictor"])

if page == "📊 Executive View":
    total = len(history_df)
    late_rate = history_df['Is_Late'].mean()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Shipments", f"{total:,}")
    k2.metric("Risk Rate", f"{late_rate:.1%}")
    k3.metric("Avg Timeline", f"{history_df['Delay_Days'].mean():.1f} Days")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⚠️ Top Risky Countries")
        risk_country = history_df.groupby('Country')['Is_Late'].mean().sort_values(ascending=False).head(10)
        st.bar_chart(risk_country)
    with c2:
        st.subheader("📦 Value vs. Risk")
        sample = history_df.sample(min(1000, len(history_df)))
        fig = px.scatter(sample, x='Line Item Value', y='Delay_Days', color='Is_Late', log_x=True,
                         color_discrete_map={True: 'red', False: 'green'})
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Risk Predictor":
    st.subheader("Interactive Risk Calculator")
    
    u_country = sorted(list(encoders['Country'].classes_))
    u_mode = sorted(list(encoders['Shipment Mode'].classes_))
    u_vendor = sorted(list(encoders['Vendor_Anonymized'].classes_))

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        country = c1.selectbox("Country", u_country)
        mode = c1.selectbox("Mode", u_mode)
        vendor = c1.selectbox("Vendor", u_vendor)
        
        qty = c2.number_input("Quantity", 100, 100000, 5000)
        val = c2.number_input("Value ($)", 100, 1000000, 50000)
        weight = c2.number_input("Weight (kg)", 10, 50000, 500)
        
        freight = c3.number_input("Freight ($)", 100, 50000, 1500)
        month = c3.slider("Month", 1, 12, 6)
        year = c3.number_input("Year", 2023, 2030, 2024)
        
        submitted = st.form_submit_button("Analyze Risk")
        
    if submitted:
        input_data = pd.DataFrame({
            'Country': [country], 'Shipment Mode': [mode], 'Vendor_Anonymized': [vendor],
            'Line Item Quantity': [qty], 'Line Item Value': [val], 'Weight (Kilograms)': [weight],
            'Freight Cost (USD)': [freight], 'Scheduled_Month': [month], 'Scheduled_Year': [year]
        })
        
        for col, le in encoders.items():
            input_data[col] = le.transform(input_data[col])
            
        prob = clf.predict_proba(input_data)[0][1]
        days = reg.predict(input_data)[0]
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.caption("AI Risk Score")
            if prob > RISK_THRESHOLD:
                st.markdown(f"<h1 style='color:red'>{prob:.0%} (HIGH RISK)</h1>", unsafe_allow_html=True)
                st.error("⚠️ Shipment flagged for potential delay.")
            else:
                st.markdown(f"<h1 style='color:green'>{prob:.0%} (LOW RISK)</h1>", unsafe_allow_html=True)
                st.success("✅ Shipment likely to arrive on time.")
        with col2:
            st.caption("Estimated Timeline")
            st.metric("Days Deviation", f"{days:.1f} Days")
