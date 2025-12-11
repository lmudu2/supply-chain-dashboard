import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIGURATION  ---
st.set_page_config(page_title="SupplyGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        h1 {margin-top: 0rem; padding-top: 0rem;}

        /* Metric Cards Styling */
        div[data-testid="stMetric"] {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #d0d0d0;
        }
        /* Force Black Text for Metrics */
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2.  Training ---
@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv('SCMS_Delivery_History_Dataset.csv', encoding='latin1')
    except FileNotFoundError:
        return None, None

    # Cleaning
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
    mode_mapping = {
        'Air': 'Air', 'Air Charter': 'Air',
        'Truck': 'Land', 'Ocean': 'Sea', 'N/A': 'Unknown'
    }
    df['Shipment Mode'] = df['Shipment Mode'].map(mode_mapping).fillna('Unknown')
    df['Product Group'] = df['Product Group'].fillna('Other')
    product_mapping = {
        'ARV': 'Medication',   # Antiretrovirals (Pills)
        'ACT': 'Medication',   # Malaria Drugs (Pills)
        'ANTM': 'Medication',  # Antimalarials (Pills)
        'HRDT': 'Test Kits',   # HIV Rapid Tests
        'MRDT': 'Test Kits'    # Malaria Rapid Tests
    }
    df['Product Group'] = df['Product Group'].replace(product_mapping)

    unique_vendors = df['Vendor'].unique()
    vendor_map = {name: f'Vendor_{i+1}' for i, name in enumerate(unique_vendors)}
    df['Vendors'] = df['Vendor'].map(vendor_map)

    # Features & Training
    feature_cols = ['Country', 'Shipment Mode', 'Vendors','Product Group',
                    'Line Item Quantity', 'Line Item Value', 'Weight (Kilograms)',
                    'Freight Cost (USD)', 'Scheduled_Month', 'Scheduled_Year']
    X = df[feature_cols].copy()

    label_encoders = {}
    for col in ['Country', 'Shipment Mode', 'Vendors','Product Group']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Train 
    clf = RandomForestClassifier(n_estimators=60, max_depth=12, random_state=42, class_weight='balanced')
    clf.fit(X, df['Is_Late'])

    reg = RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42)
    reg.fit(X, df['Delay_Days'])

    return clf, reg, label_encoders, df

# Loading model
with st.spinner("🧠 Booting AI..."):
    clf, reg, encoders, history_df = load_and_train_model()

if clf is None:
    st.error("❌ CSV Missing!")
    st.stop()

unique_values = {col: history_df[col].unique() for col in ['Country', 'Shipment Mode', 'Vendors', 'Product Group']}

# --- 3. HEADER & GLOBAL FILTERS  ---
c1, c2 = st.columns([2, 2])
with c1:
    st.title("Supplier Risk Analysis")
    # st.caption("AI-Powered Detection of Logistics Bottlenecks")

with c2:
    # Top-Bar Filters
    f1, f2 = st.columns(2)
    with f1:
        # Date Filter
        min_date = history_df['Scheduled Delivery Date'].min()
        max_date = history_df['Scheduled Delivery Date'].max()
        start_date, end_date = st.date_input("📅 Date Range", [min_date, max_date])
    with f2:
        # MODE FILTER (Switched from Country)
        selected_mode = st.selectbox("Shipment Mode", ["All"] + sorted(history_df['Shipment Mode'].unique().tolist()))

# --- APPLY FILTERS ---
filtered_df = history_df.copy()
mask = (filtered_df['Scheduled Delivery Date'].dt.date >= start_date) & (filtered_df['Scheduled Delivery Date'].dt.date <= end_date)
filtered_df = filtered_df.loc[mask]

if selected_mode != "All":
    filtered_df = filtered_df[filtered_df['Shipment Mode'] == selected_mode]

# --- 4. TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Executive View", "🔮 Risk Simulator", "📋 Data Explorer"])

# === TAB 1: DASHBOARD ===
with tab1:
    if filtered_df.empty:
        st.warning("No data matches these filters.")
    else:
        # Row 1: KPIs
        total = len(filtered_df)
        late_rate = filtered_df['Is_Late'].mean()
        avg_delay = filtered_df['Delay_Days'].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Shipments", f"{total:,}")
        k2.metric("On-Time Rate", f"{100 - (late_rate*100):.1f}%")
        k3.metric("High Risk Rate", f"{late_rate:.1%}", delta_color="inverse")
        k4.metric("Avg Timeline", f"{avg_delay:.1f} Days")

        # Row 2: Charts 
        row2_1, row2_2 = st.columns(2)

        with row2_1:
            # BAR CHART: Top Risky COUNTRIES (Switched back from Vendors)
            st.markdown("##### ⚠️ Top Risky Countries")
            country_risk = filtered_df.groupby('Country')['Is_Late'].mean().reset_index()
            # Sort and take top 8
            country_risk = country_risk.sort_values('Is_Late', ascending=False).head(8)

            fig_bar = px.bar(country_risk, x='Is_Late', y='Country', orientation='h',
                             title="", color='Is_Late', color_continuous_scale='Reds')
            fig_bar.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Probability of Delay")
            st.plotly_chart(fig_bar, use_container_width=True)

        with row2_2:
            # SCATTER CHART
            # st.markdown("##### 📦 Cost vs. Risk")
            # sample = filtered_df.sample(min(500, len(filtered_df)))
            # fig_scatter = px.scatter(sample, x='Line Item Value', y='Delay_Days', color='Is_Late', size='Line Item Quantity',
            #                          title="", color_discrete_map={True: 'red', False: 'green'}, log_x=True)
            # fig_scatter.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            # st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("##### 💰 Risk Rate by Shipment Value")
            chart_df = filtered_df.copy()
            bins = [-1, 10000, 50000, 250000, float('inf')]
            labels = ['Low (<$10k)', 'Medium ($10k-50k)', 'High ($50k-250k)', 'Very High (>$250k)']
            chart_df['Value_Bin'] = pd.cut(chart_df['Line Item Value'], bins=bins, labels=labels)

            risk_by_value = chart_df.groupby('Value_Bin', observed=False)['Is_Late'].mean().reset_index()

            fig_val = px.bar(
                risk_by_value, x='Value_Bin', y='Is_Late', text_auto='.1%',
                color='Is_Late', color_continuous_scale='Reds',
                labels={'Value_Bin': 'Value Tier', 'Is_Late': 'Risk Probability'}
            )
            # ROBUST LAYOUT
            fig_val.update_layout(yaxis_tickformat='.0%', height=300, margin=dict(l=50, r=20, t=20, b=50))
            st.plotly_chart(fig_val, use_container_width=True)

# === TAB 2: SIMULATOR ===
with tab2:
    # st.markdown("##### 🔮 Risk Calculator")

    with st.form("sim_form"):
        # Compact 3-column layout
        c1, c2, c3 = st.columns(3)
        with c1:
            country_in = st.selectbox("Destination", sorted(unique_values['Country']))
            mode_in = st.selectbox("Mode", sorted(unique_values['Shipment Mode']))
            vendor_in = st.selectbox("Vendor", sorted(unique_values['Vendors']))
        with c2:
            qty_in = st.number_input("Qty", 100, 100000, 5000)
            val_in = st.number_input("Value ($)", 1000, 1000000, 50000)
            weight_in = st.number_input("Weight (kg)", 10, 50000, 500)
        with c3:
            freight_in = st.number_input("Freight ($)", 100, 50000, 1500)
            month_in = st.slider("Month", 1, 12, 6)
            year_in = st.number_input("Year", 2023, 2030, 2024)

        # Submit Button
        submitted = st.form_submit_button("🚀 Risk Prediction", use_container_width=True)

    if submitted:
        # Prepare Data
        input_data = pd.DataFrame({
            'Country': [country_in], 'Shipment Mode': [mode_in], 'Vendors': [vendor_in],
            'Line Item Quantity': [qty_in], 'Line Item Value': [val_in], 'Weight (Kilograms)': [weight_in],
            'Freight Cost (USD)': [freight_in], 'Scheduled_Month': [month_in], 'Scheduled_Year': [year_in]
        })

        for col, le in encoders.items():
            val = str(input_data[col][0])
            input_data[col] = le.transform([val]) if val in le.classes_ else le.transform([le.classes_[0]])

        prob = clf.predict_proba(input_data)[0][1]
        days = reg.predict(input_data)[0]

        # Results
        st.divider()
        res1, res2, res3 = st.columns([1, 2, 1])
        with res1:
            st.caption("AI Probability")
            st.metric("Risk Score", f"{prob:.0%}")
        with res2:
            st.caption("Status")
            if prob > 0.25:
                st.error(f"🔴 HIGH RISK (Score > 25%)")
            else:
                st.success(f"🟢 LOW RISK (Safe)")
        with res3:
            st.caption("Timeline")
            st.metric("Est. Deviation", f"{days:.1f} Days")

# === TAB 3: DATA EXPLORER (Hidden unless needed) ===
with tab3:
    st.dataframe(filtered_df[['Scheduled Delivery Date', 'Country', 'Vendors', 'Shipment Mode', 'Line Item Value', 'Delay_Days', 'Is_Late']])
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", data=csv, file_name="supply_chain_data.csv", mime="text/csv")
