import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
from src.database import init_db, save_to_db

def generate_synthetic_data(num_assets=500):
    np.random.seed(42)
    random.seed(42)
    
    # 2026 Reference Date
    current_date = datetime(2026, 6, 19)
    
    categories = {
        "Manufacturing Equipment": [
            "CNC Wheel Lathe", "CNC Milling Machine", "CNC Turning Center",
            "Welding Robot", "Hydraulic Press", "Compressor Unit",
            "Paint Booth Ventilation System", "Material Handling Crane"
        ],
        "Locomotive Subsystems": [
            "Traction Motor", "Air Compressor", "Alternator",
            "Cooling System", "Brake System", "Wheel Assembly",
            "Axle Bearing Assembly", "Diesel Engine Auxiliary System"
        ]
    }
    
    all_categories = categories["Manufacturing Equipment"] + categories["Locomotive Subsystems"]
    
    data = []
    
    for i in range(1, num_assets + 1):
        m_id = f"LOC-EQ-{i:03d}"
        category_group = "Manufacturing Equipment" if i % 2 == 0 else "Locomotive Subsystems"
        m_category = random.choice(categories[category_group])
        m_name = f"{m_category} {random.randint(100, 999)}"
        
        power_rating = float(np.round(np.random.uniform(15.0, 150.0), 1))
        avg_daily_hours = float(np.round(np.random.uniform(6.0, 18.0), 1))
        
        age_days = int(np.random.randint(180, 4500))
        installation_dt = current_date - timedelta(days=age_days)
        total_runtime = float(np.round(age_days * avg_daily_hours * np.random.uniform(0.7, 0.95), 1))
        
        breakdowns = int(np.random.poisson(lam=age_days / 500))
        days_since_maint = int(np.random.randint(5, 300))
        last_maint_dt = current_date - timedelta(days=days_since_maint)
        
        criticality = random.choice(["Low", "Medium", "High", "Critical"])
        if m_category in ["Traction Motor", "Brake System", "CNC Wheel Lathe"]:
            criticality = "Critical" if random.random() > 0.15 else "High"
            
        base_temp = 65.0 if m_category in ["Traction Motor", "Diesel Engine Auxiliary System"] else 50.0
        wear_factor = (days_since_maint / 150.0) + (breakdowns * 0.2)
        
        avg_temp = float(np.round(base_temp + (np.random.normal(5, 2) * wear_factor), 1))
        avg_vib = float(np.round(1.5 + (np.random.normal(0.4, 0.1) * wear_factor * 1.5), 2))
        avg_current = float(np.round((power_rating * 1.5) + np.random.normal(0, 3), 1))
        
        avg_temp = max(35.0, min(avg_temp, 125.0))
        avg_vib = max(0.2, min(avg_vib, 12.0))
        avg_current = max(5.0, avg_current)
        
        maint_cost = float(np.round((breakdowns * 2500.0) + (total_runtime * 0.15) + np.random.uniform(500, 2000), 2))
        
        data.append({
            "machine_id": m_id,
            "machine_name": m_name,
            "machine_category": m_category,
            "installation_date": installation_dt.strftime("%Y-%m-%d"),
            "machine_age_days": age_days,
            "power_rating_kw": power_rating,
            "avg_op_hours_per_day": avg_daily_hours,
            "total_runtime_hours": total_runtime,
            "breakdown_count": breakdowns,
            "days_since_last_maint": days_since_maint,
            "last_maint_date": last_maint_dt.strftime("%Y-%m-%d"),
            "criticality_level": criticality,
            "avg_temp_c": avg_temp,
            "avg_vibration_mms": avg_vib,
            "avg_current_amp": avg_current,
            "maint_cost_history": maint_cost
        })
        
    df = pd.DataFrame(data)
    init_db()
    save_to_db(df)
    print(f"Database populated with {len(df)} realistic records.")
    return df

if __name__ == "__main__":
    generate_synthetic_data()