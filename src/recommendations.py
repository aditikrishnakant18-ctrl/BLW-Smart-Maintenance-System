from datetime import datetime, timedelta

def generate_recommendations(row):
    recs = []
    explanations = []
    
    if row['avg_temp_c'] > 85.0:
        recs.append("CRITICAL: Inspect auxiliary cooling system, radiator fins, and check coolant level.")
        explanations.append(f"Operating temperature ({row['avg_temp_c']}°C) exceeds safety limit of 85°C.")
    elif row['avg_temp_c'] > 72.0:
        recs.append("WARNING: Schedule cooling duct wash during next scheduled shift.")
        explanations.append(f"Elevated temperature ({row['avg_temp_c']}°C) indicates minor thermal degradation.")
        
    if row['avg_vibration_mms'] > 4.5:
        recs.append("CRITICAL: Immediate laser alignment and bearing lubrication required.")
        explanations.append(f"Structural vibration ({row['avg_vibration_mms']} mm/s) exceeds maximum ISO velocity limits.")
    elif row['avg_vibration_mms'] > 2.8:
        recs.append("WARNING: Perform structural bolt torque validation and inspect shock mounts.")
        explanations.append(f"Vibration level ({row['avg_vibration_mms']} mm/s) is entering warning boundary.")
        
    if row['total_runtime_hours'] > 25000:
       recs.append(
        "LIFECYCLE ALERT: Equipment has crossed 25,000 operating hours. Engineering review recommended."
    )
       explanations.append(
        f"Total runtime ({row['total_runtime_hours']:.1f} hrs) exceeds lifecycle threshold."
    )

    elif row['total_runtime_hours'] > 15000:
       recs.append(
        "PREVENTIVE: High cumulative operating hours detected. Monitor wear components."
    )
       explanations.append(
        f"Total runtime ({row['total_runtime_hours']:.1f} hrs) indicates advanced wear."
    )
       
    if row['days_since_last_maint'] > 120:
        recs.append("COMPLIANCE: Perform physical lubrication, seal inspections, and filter swap.")
        explanations.append(f"Asset is past preventive interval: {row['days_since_last_maint']} days since last service.")
        
    if not recs:
        recs.append("Maintain standard operational schedule. No anomalies detected.")
        explanations.append("All metrics fall within nominal boundaries.")
        
    return recs, explanations

def calculate_scheduled_date(last_maint_str: str, rul_days: float) -> str:
    try:
        # Maintain reference context of June 2026
        run_date = datetime(2026, 6, 19)
        recommended_dt = run_date + timedelta(days=max(1, int(rul_days)))
        return recommended_dt.strftime("%d %B %Y")
    except Exception:
        return "N/A"