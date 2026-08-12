import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from src.database import fetch_all_equipment, fetch_machine_by_id
from src.health_engine import calculate_health_score
from src.recommendations import generate_recommendations, calculate_scheduled_date

st.set_page_config(page_title="BLW Smart Maintenance System", layout="wide")

@st.cache_resource
def load_ml_resources():
    try:
        scaler = joblib.load(os.path.join("models", "scaler.joblib"))
        rul_model = joblib.load(os.path.join("models", "rul_regressor.joblib"))
        pri_model = joblib.load(os.path.join("models", "priority_clf.joblib"))
        return scaler, rul_model, pri_model
    except FileNotFoundError:
        return None, None, None

scaler, rul_model, pri_model = load_ml_resources()

df_all = fetch_all_equipment()
if df_all.empty:
    st.error("Critical System Warning: SQL Database contains no equipment records. Run generator.py first.")
    st.stop()

df_all['health_score'] = df_all.apply(calculate_health_score, axis=1)

def get_priority_color(priority):
    colors = {
        "Critical": "#D32F2F",
        "High": "#F57C00",
        "Medium": "#FBC02D",
        "Low": "#388E3C"
    }
    return colors.get(priority, "#9E9E9E")

st.markdown(
    """
    <div style="background-color:#1E3A8A;padding:20px;border-radius:8px;margin-bottom:25px">
        <h1 style="color:white;margin:0;font-size:32px;">AI-BASED PREDICTIVE MAINTENANCE & ASSET HEALTH MONITORING SYSTEM</h1>
        <p style="color:#E2E8F0;margin:5px 0 0 0;font-size:16px;">Machine Learning Enabled Predictive Maintenance Platform for Banaras Locomotive Works (BLW)</p>
    </div>
    """, 
    unsafe_allow_html=True
)

tab_home, tab_predict = st.tabs(["Facility Fleet Overview", "Asset Diagnostics & Predictions"])

# ==================== TAB 1: HOME PAGE ====================
with tab_home:
    st.subheader("Operational Plant Dashboard")
    
    total_assets = len(df_all)
    avg_health_score = df_all['health_score'].mean()
    
    critical_assets_df = df_all[df_all['health_score'] < 50]
    high_assets_df = df_all[(df_all['health_score'] >= 50) & (df_all['health_score'] < 70)]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Monitored Assets", total_assets)
    with col2:
        st.metric("Average Fleet Health", f"{avg_health_score:.1f}%")
    with col3:
        st.metric("CRITICAL Status Assets (<50%)", len(critical_assets_df))
    with col4:
        st.metric("HIGH Risk Status Assets (50-70%)", len(high_assets_df))
        
    st.write("---")
    st.subheader("Critical and High Risk Assets Requiring Immediate Inspection")
    
    flagged_assets = df_all[df_all['health_score'] < 70].sort_values(by="health_score")
    if not flagged_assets.empty:
        display_columns = [
            "machine_id", "machine_name", "machine_category", 
            "criticality_level", "health_score", "days_since_last_maint", "avg_temp_c", "avg_vibration_mms"
        ]
        st.dataframe(
            flagged_assets[display_columns].rename(columns={
                "machine_id": "ID", "machine_name": "Name", "machine_category": "Category",
                "criticality_level": "Criticality", "health_score": "Health Score",
                "days_since_last_maint": "Days Since Maintenance", "avg_temp_c": "Temp (°C)", "avg_vibration_mms": "Vibration (mm/s)"
            }), use_container_width=True
        )
    else:
        st.info("No assets currently fall below nominal safety standards.")

# ==================== TAB 2: DETAILED DIAGNOSTICS ====================
with tab_predict:
    st.subheader("Asset Diagnostics & Schedule Forecasting Panel")
    
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        selected_category = st.selectbox("Filter Category", ["All"] + list(df_all['machine_category'].unique()))
    
    if selected_category != "All":
        filtered_df = df_all[df_all['machine_category'] == selected_category]
    else:
        filtered_df = df_all
        
    equipment_options = [f"{row['machine_id']} - {row['machine_name']}" for _, row in filtered_df.iterrows()]
    
    with col_sel2:
        selected_eq_str = st.selectbox("Select Equipment ID / Name", equipment_options)
        
    if selected_eq_str:
        selected_id = selected_eq_str.split(" - ")[0]
        eq_data = fetch_machine_by_id(selected_id)
        
        st.markdown(f"### Operational Profile: {eq_data['machine_name']} (`{eq_data['machine_id']}`)")
        st.write("---")
        st.subheader("Live Machine Simulation")

        colA, colB = st.columns(2)

        with colA:

            live_temp = st.number_input(
                "Current Temperature (°C)",
                value=float(eq_data["avg_temp_c"])
            )

            live_vibration = st.number_input(
                "Current Vibration (mm/s)",
                value=float(eq_data["avg_vibration_mms"])
            )

        with colB:

            live_current = st.number_input(
                "Current Drawn Current (A)",
                value=float(eq_data["avg_current_amp"])
            )

            live_days = st.number_input(
                "Days Since Last Maintenance",
                value=int(eq_data["days_since_last_maint"])
            )

        eq_data["avg_temp_c"] = live_temp
        eq_data["avg_vibration_mms"] = live_vibration
        eq_data["avg_current_amp"] = live_current
        eq_data["days_since_last_maint"] = live_days
        
        st.subheader("Failure Simulation")

        f1, f2, f3 = st.columns(3)

        with f1:

            if st.button("Bearing Failure"):
               eq_data["avg_vibration_mms"] += 3

        with f2:

            if st.button("Cooling Failure"):
               eq_data["avg_temp_c"] += 20

        with f3:

            if st.button("Lubrication Failure"):
                eq_data["avg_temp_c"] += 10
                eq_data["avg_vibration_mms"] += 2
        
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.markdown(f"**Category:** {eq_data['machine_category']}")
        stat2.markdown(f"**Asset Criticality:** {eq_data['criticality_level']}")
        stat3.markdown(f"**Power Rating:** {eq_data['power_rating_kw']} kW")
        stat4.markdown(f"**Installed:** {eq_data['installation_date']}")
        
        st.write("---")
        
        features = [
            "machine_age_days", "total_runtime_hours", "breakdown_count", 
            "days_since_last_maint", "avg_temp_c", "avg_vibration_mms", "avg_current_amp"
        ]
        
        input_values = np.array([[eq_data[feat] for feat in features]])
        
        if scaler and rul_model and pri_model:
            input_scaled = scaler.transform(input_values)
            predicted_rul = float(rul_model.predict(input_scaled)[0])
            predicted_priority = str(pri_model.predict(input_scaled)[0])
        else:
            st.warning("Model binaries not found. Using calculated mathematical estimates.")
            h_sc = calculate_health_score(eq_data)
            predicted_rul = (h_sc / 100.0) * 120.0
            if h_sc < 50: predicted_priority = "Critical"
            elif h_sc < 70: predicted_priority = "High"
            elif h_sc < 85: predicted_priority = "Medium"
            else: predicted_priority = "Low"
            
        health_score_val = calculate_health_score(eq_data)
        recommended_date = calculate_scheduled_date(eq_data['last_maint_date'], predicted_rul)
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(
                f"""
                <div style="background-color:#F8FAFC;border-left:5px solid #2563EB;padding:15px;border-radius:4px;text-align:center">
                    <p style="margin:0;font-size:14px;color:#64748B;font-weight:600;">HEALTH SCORE</p>
                    <h2 style="margin:5px 0 0 0;color:#1E293B;font-size:36px;">{health_score_val}%</h2>
                </div>
                """, unsafe_allow_html=True
            )
            
        with m2:
            st.markdown(
                f"""
                <div style="background-color:#F8FAFC;border-left:5px solid {get_priority_color(predicted_priority)};padding:15px;border-radius:4px;text-align:center">
                    <p style="margin:0;font-size:14px;color:#64748B;font-weight:600;">MAINTENANCE PRIORITY</p>
                    <h2 style="margin:5px 0 0 0;color:{get_priority_color(predicted_priority)};font-size:36px;">{predicted_priority}</h2>
                </div>
                """, unsafe_allow_html=True
            )
            
        with m3:
            st.markdown(
                f"""
                <div style="background-color:#F8FAFC;border-left:5px solid #0D9488;padding:15px;border-radius:4px;text-align:center">
                    <p style="margin:0;font-size:14px;color:#64748B;font-weight:600;">REMAINING USEFUL LIFE</p>
                    <h2 style="margin:5px 0 0 0;color:#0F766E;font-size:36px;">{int(predicted_rul)} Days</h2>
                </div>
                """, unsafe_allow_html=True
            )
            
        with m4:
            st.markdown(
                f"""
                <div style="background-color:#F8FAFC;border-left:5px solid #7C3AED;padding:15px;border-radius:4px;text-align:center">
                    <p style="margin:0;font-size:14px;color:#64748B;font-weight:600;">RECOMMENDED DATE</p>
                    <h2 style="margin:5px 0 0 0;color:#6D28D9;font-size:22px;padding-top:10px;">{recommended_date}</h2>
                </div>
                """, unsafe_allow_html=True
            )
            
        st.markdown("#### Real-time Physical Telemetry vs. Baseline Metrics")
        
        tele_col1, tele_col2, tele_col3, tele_col4 = st.columns(4)
        tele_col1.metric("Current Temperature", f"{eq_data['avg_temp_c']} °C", delta=f"{eq_data['avg_temp_c'] - 65.0:.1f} °C vs. Base", delta_color="inverse")
        tele_col2.metric("Vibration Speed", f"{eq_data['avg_vibration_mms']} mm/s", delta=f"{eq_data['avg_vibration_mms'] - 1.50:.2f} mm/s vs. Base", delta_color="inverse")
        tele_col3.metric("Current Drawn", f"{eq_data['avg_current_amp']} A")
        tele_col4.metric("Days Since Last Maintenance", f"{eq_data['days_since_last_maint']} Days")
        
        st.write("---")
        rec_col, exp_col = st.columns(2)
        
        recs, explanations = generate_recommendations(eq_data)
        
        with rec_col:
            st.markdown("####  Engineering Action Recommendations")
            for rec in recs:
                if "CRITICAL" in rec:
                    st.error(rec)
                elif "WARNING" in rec:
                    st.warning(rec)
                else:
                    st.success(rec)
                    
        with exp_col:
            st.markdown("####  Anomaly Explanations & Feature Importance")
            for exp in explanations:
                st.info(exp)
                
            if scaler and rul_model:
                importances = rul_model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": ["Age", "Total Runtime", "Breakdowns", "Last Maint. Interval", "Temp", "Vibration", "Current"],
                    "Weight": importances
                }).sort_values(by="Weight", ascending=False)
                
                st.write("##### Critical Prediction Drivers (Feature Weights)")
                st.dataframe(importance_df, use_container_width=True, hide_index=True)