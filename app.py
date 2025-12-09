import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Unified Supply Chain Risk Dashboard", page_icon="🚚", layout="wide")

# --- 2.Trains data ---
@st.cache_resource
def load_and_train_model():
    # A. Load Data
    try:
        df = pd.read_csv('SCMS_Delivery_History_Dataset.csv', encoding='latin1')
    except FileNotFoundError:
        return None, None

    # B. Cleanup & Preprocessing
    if 'ID' in df.columns: df.rename(columns={'ID': 'id'}, inplace=True)
    elif 'ï»¿ID' in df.columns: df.rename(columns={'ï»¿ID': 'id'}, inplace=True)

    df['Scheduled Delivery Date'] = pd.to_datetime(df['Scheduled Delivery Date'], errors='coerce')
    df['Delivered to Client Date'] = pd.to_datetime(df['Delivered to Client Date'], errors='coerce')
    df = df.dropna(subset=['Scheduled Delivery Date', 'Delivered to Client Date'])
    
    df['Delay_Days'] = (df['Delivered to Client Date'] - df['Scheduled Delivery Date']).dt.days
    df['Is_Late'] = (df['Delay_Days'] > 0).astype(int)

    # Numeric Cleaning
    for col in ['Weight (Kilograms)', 'Freight Cost (USD)', 'Line Item Value', 'Line Item Quantity']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Shipment Mode'] = df['Shipment Mode'].fillna('Unknown')
    df['Scheduled_Month'] = df['Scheduled Delivery Date'].dt.month
    df['Scheduled_Year'] = df['Scheduled Delivery Date'].dt.year

    # Vendors
    unique_vendors = df['Vendor'].unique()
    vendor_map = {name: f'Vendor_{i+1}' for i, name in enumerate(unique_vendors)}
    df['Vendors'] = df['Vendor'].map(vendor_map)

    # C. Features & Encoding
    feature_cols = ['Country', 'Shipment Mode', 'Vendors', 
                    'Line Item Quantity', 'Line Item Value', 'Weight (Kilograms)', 
                    'Freight Cost (USD)', 'Scheduled_Month', 'Scheduled_Year']
    X = df[feature_cols].copy()

    label_encoders = {}
    for col in ['Country', 'Shipment Mode', 'Vendors']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # D. Train Models (Paranoid Mode Enabled)
    # Using n_estimators=60 to keep it fast
    clf = RandomForestClassifier(n_estimators=60, max_depth=12, random_state=42, class_weight='balanced')
    clf.fit(X, df['Is_Late'])
    
    reg = RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42)
    reg.fit(X, df['Delay_Days'])

    model_artifacts = {
        'classifier': clf,
        'regressor': reg,
        'encoders': label_encoders,
        'feature_cols': feature_cols,
        'unique_values': {col: df[col].astype(str).unique().tolist() for col in ['Country', 'Shipment Mode', 'Vendors']}
    }
    
    return model_artifacts, df


with st.spinner("🧠 Initializing Model..."):
    model_artifacts, history_df = load_and_train_model()

if model_artifacts is None:
    st.error("❌ Error: 'SCMS_Delivery_History_Dataset.csv' not found. Please upload it to GitHub!")
    st.stop()

# Unpack for use
clf = model_artifacts['classifier']
reg = model_artifacts['regressor']
encoders = model_artifacts['encoders']
feature_cols = model_artifacts['feature_cols']
unique_values = model_artifacts['unique_values']

# --- 3. SIDEBAR ---
st.sidebar.title("Supply Chain AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 Executive Dashboard", "🔮 Risk Predictor"])

# --- 4. PAGE: EXECUTIVE DASHBOARD ---
if page == "📊 Executive Dashboard":
    st.title("📊 Unified Supply Chain Risk View")

    if history_df.empty:
        st.warning("Upload the dataset to see analytics.")
    else:
        # KPIs
        total_shipments = len(history_df)
        late_shipments = history_df['Is_Late'].sum()
        avg_delay = history_df['Delay_Days'].mean()
        late_rate = (late_shipments / total_shipments) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Shipments", f"{total_shipments:,}")
        col2.metric("On-Time Rate", f"{100 - late_rate:.1f}%")
        col3.metric("High Risk (Late) Rate", f"{late_rate:.1f}%", delta_color="inverse")
        col4.metric("Avg Delay", f"{avg_delay:.1f} Days")
        st.markdown("---")

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("⚠️ Risk by Country")
            country_risk = history_df.groupby('Country')['Is_Late'].mean().reset_index()
            country_risk = country_risk.sort_values('Is_Late', ascending=False).head(10)
            fig_country = px.bar(country_risk, x='Is_Late', y='Country', orientation='h',
                                 title="Top 10 High-Risk Destinations", color='Is_Late', color_continuous_scale='Reds')
            st.plotly_chart(fig_country, use_container_width=True)

        with c2:
            st.subheader("📦 Value vs. Risk")
            sample_df = history_df.sample(min(1000, len(history_df)))
            fig_scatter = px.scatter(
                sample_df,
                x='Line Item Value',
                y='Delay_Days',
                color='Is_Late',
                size='Line Item Quantity',
                title="High Value Shipment Analysis",
                log_x=True,
                color_discrete_map={True: 'red', False: 'green'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Feature Importance
        st.subheader("🔍 What Drives Risk?")
        importances = clf.feature_importances_
        feature_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
        feature_df = feature_df.sort_values('Importance', ascending=True)
        fig_imp = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                         title="Key Drivers", color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig_imp, use_container_width=True)

# --- 5. PAGE: RISK PREDICTOR ---
elif page == "🔮 Risk Predictor":
    st.title("🔮 Interactive Risk Scorer")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            country_in = st.selectbox("Destination", sorted(unique_values['Country']))
            mode_in = st.selectbox("Shipment Mode", sorted(unique_values['Shipment Mode']))
            vendor_in = st.selectbox("Vendor", sorted(unique_values['Vendors']))
        with c2:
            qty_in = st.number_input("Quantity", 1, 100000, 1000)
            val_in = st.number_input("Value ($)", 0.0, 1000000.0, 5000.0)
            weight_in = st.number_input("Weight (kg)", 0.0, 50000.0, 100.0)
        with c3:
            freight_in = st.number_input("Freight ($)", 0.0, 50000.0, 500.0)
            month_in = st.slider("Scheduled Month", 1, 12, 6)
            year_in = st.number_input("Scheduled Year", 2000, 2030, 2023)

        submit = st.form_submit_button("Calculate Risk")

    if submit:
        input_data = pd.DataFrame({
            'Country': [country_in], 'Shipment Mode': [mode_in], 'Vendors': [vendor_in],
            'Line Item Quantity': [qty_in], 'Line Item Value': [val_in], 'Weight (Kilograms)': [weight_in],
            'Freight Cost (USD)': [freight_in], 'Scheduled_Month': [month_in], 'Scheduled_Year': [year_in]
        })

        for col, le in encoders.items():
            val = str(input_data[col][0])
            input_data[col] = le.transform([val]) if val in le.classes_ else le.transform([le.classes_[0]])

        risk_prob = clf.predict_proba(input_data)[0][1]
        delay_est = reg.predict(input_data)[0]

        st.divider()
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            st.caption("Risk Score")
            if risk_prob > 0.25:
                st.markdown(f"<h1 style='color:red'>{risk_prob:.0%}</h1>", unsafe_allow_html=True)
                st.error("HIGH RISK")
            else:
                st.markdown(f"<h1 style='color:green'>{risk_prob:.0%}</h1>", unsafe_allow_html=True)
                st.success("LOW RISK")

        with col_res2:
            st.caption("Operational Impact")
            st.metric("Estimated Timeline", f"{delay_est:.1f} Days vs Schedule")
