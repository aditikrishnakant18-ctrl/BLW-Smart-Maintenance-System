"""
Synthetic Dataset Generation for BLW Smart Maintenance System
Generates realistic industrial maintenance data for locomotive manufacturing equipment.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# Equipment Master List
# ─────────────────────────────────────────────
EQUIPMENT_CATALOG = [
    # Manufacturing Equipment
    {"name": "CNC Wheel Lathe",              "category": "Manufacturing", "power_kw": 45,  "criticality": "High",     "design_life_days": 3650},
    {"name": "CNC Milling Machine",          "category": "Manufacturing", "power_kw": 30,  "criticality": "High",     "design_life_days": 3650},
    {"name": "CNC Turning Center",           "category": "Manufacturing", "power_kw": 22,  "criticality": "High",     "design_life_days": 3650},
    {"name": "Welding Robot",                "category": "Manufacturing", "power_kw": 18,  "criticality": "Medium",   "design_life_days": 2920},
    {"name": "Hydraulic Press",              "category": "Manufacturing", "power_kw": 55,  "criticality": "High",     "design_life_days": 5475},
    {"name": "Compressor Unit",              "category": "Manufacturing", "power_kw": 37,  "criticality": "Medium",   "design_life_days": 3650},
    {"name": "Paint Booth Ventilation",      "category": "Manufacturing", "power_kw": 15,  "criticality": "Low",      "design_life_days": 2555},
    {"name": "Material Handling Crane",      "category": "Manufacturing", "power_kw": 75,  "criticality": "Critical", "design_life_days": 7300},
    # Locomotive Subsystems
    {"name": "Traction Motor",               "category": "Locomotive",    "power_kw": 350, "criticality": "Critical", "design_life_days": 4380},
    {"name": "Air Compressor",               "category": "Locomotive",    "power_kw": 22,  "criticality": "High",     "design_life_days": 3650},
    {"name": "Alternator",                   "category": "Locomotive",    "power_kw": 110, "criticality": "Critical", "design_life_days": 4380},
    {"name": "Cooling System",               "category": "Locomotive",    "power_kw": 18,  "criticality": "High",     "design_life_days": 3285},
    {"name": "Brake System",                 "category": "Locomotive",    "power_kw": 12,  "criticality": "Critical", "design_life_days": 2920},
    {"name": "Wheel Assembly",               "category": "Locomotive",    "power_kw": 0,   "criticality": "Critical", "design_life_days": 5475},
    {"name": "Axle Bearing Assembly",        "category": "Locomotive",    "power_kw": 0,   "criticality": "Critical", "design_life_days": 4015},
    {"name": "Diesel Engine Auxiliary",      "category": "Locomotive",    "power_kw": 85,  "criticality": "High",     "design_life_days": 3650},
]

CRITICALITY_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# Normal operating ranges per equipment type
SENSOR_PROFILES = {
    "Manufacturing": {
        "temp_mean": 65,  "temp_std": 12,
        "vib_mean": 2.8,  "vib_std": 0.9,
        "curr_mean": 0.82,"curr_std": 0.10,
    },
    "Locomotive": {
        "temp_mean": 78,  "temp_std": 15,
        "vib_mean": 3.5,  "vib_std": 1.2,
        "curr_mean": 0.85,"curr_std": 0.12,
    },
}

def generate_machine_id(idx, category):
    prefix = "MFG" if category == "Manufacturing" else "LOC"
    return f"BLW-{prefix}-{idx:04d}"

def compute_rul(age_days, design_life_days, total_runtime_hours, breakdowns, temp, vibration, criticality):
    """Estimate Remaining Useful Life in days using degradation physics."""
    age_factor          = age_days / design_life_days
    runtime_factor      = min(total_runtime_hours / (design_life_days * 16), 1.0)
    temp_stress         = max(0, (temp - 60) / 40)
    vib_stress          = max(0, (vibration - 2.0) / 4.0)
    breakdown_penalty   = breakdowns * 0.04
    crit_weight         = CRITICALITY_MAP[criticality] / 4.0

    degradation = (
        0.30 * age_factor +
        0.25 * runtime_factor +
        0.20 * temp_stress +
        0.15 * vib_stress +
        0.10 * breakdown_penalty
    ) * (0.85 + 0.15 * crit_weight)

    degradation = min(degradation, 1.0)
    remaining_life_fraction = max(0.0, 1.0 - degradation)
    rul = remaining_life_fraction * design_life_days * 0.6
    noise = np.random.normal(0, rul * 0.05)
    return max(1, round(rul + noise))

def compute_priority(rul, health_score, criticality, days_since_maintenance):
    """Classify maintenance priority."""
    crit_val = CRITICALITY_MAP[criticality]
    if rul < 15 or health_score < 30 or (crit_val == 4 and rul < 30):
        return "Critical"
    elif rul < 45 or health_score < 50 or days_since_maintenance > 180:
        return "High"
    elif rul < 90 or health_score < 70 or days_since_maintenance > 120:
        return "Medium"
    else:
        return "Low"

def compute_health_score(age_days, design_life_days, total_runtime_hours,
                         temp, vibration, breakdowns, days_since_maintenance):
    """Composite health score 0–100."""
    age_score       = max(0, 100 - (age_days / design_life_days) * 100)
    runtime_score   = max(0, 100 - (total_runtime_hours / (design_life_days * 16)) * 100)
    temp_score      = max(0, 100 - max(0, (temp - 60) / 0.5))
    vib_score       = max(0, 100 - max(0, (vibration - 2.0) * 20))
    breakdown_score = max(0, 100 - breakdowns * 12)
    maint_score     = max(0, 100 - (days_since_maintenance / 180) * 50)

    score = (
        0.20 * age_score +
        0.20 * runtime_score +
        0.20 * temp_score +
        0.15 * vib_score +
        0.15 * breakdown_score +
        0.10 * maint_score
    )
    noise = np.random.normal(0, 1.5)
    return round(min(100, max(0, score + noise)), 1)

def generate_dataset(n_machines=500, obs_per_machine=10):
    records = []
    machine_id_counter = 1
    base_date = datetime(2020, 1, 1)

    for _ in range(n_machines):
        eq = random.choice(EQUIPMENT_CATALOG)
        profile = SENSOR_PROFILES[eq["category"]]

        install_date = base_date + timedelta(days=random.randint(0, 1825))
        age_days = (datetime(2026, 6, 1) - install_date).days

        machine_id = generate_machine_id(machine_id_counter, eq["category"])
        machine_id_counter += 1

        avg_ops_hours = round(random.uniform(6, 20), 1)
        total_runtime = round(age_days * avg_ops_hours * random.uniform(0.7, 1.0))

        for obs in range(obs_per_machine):
            obs_date = datetime(2026, 1, 1) + timedelta(days=obs * 15)

            # Sensor readings with realistic drift over observations
            drift = obs / obs_per_machine
            temp      = round(np.random.normal(profile["temp_mean"] + drift * 8, profile["temp_std"]), 1)
            vibration = round(np.random.normal(profile["vib_mean"] + drift * 0.5, profile["vib_std"]), 2)
            current   = round(np.random.normal(profile["curr_mean"] + drift * 0.05, profile["curr_std"]), 3)
            current   = max(0.3, min(1.5, current))

            breakdowns          = random.randint(0, 8)
            days_since_maint    = random.randint(0, 365)
            last_maint_date     = obs_date - timedelta(days=days_since_maint)
            maint_cost_history  = round(random.uniform(5000, 150000), 0)

            health  = compute_health_score(age_days, eq["design_life_days"], total_runtime,
                                           temp, vibration, breakdowns, days_since_maint)
            rul     = compute_rul(age_days, eq["design_life_days"], total_runtime,
                                  breakdowns, temp, vibration, eq["criticality"])
            priority = compute_priority(rul, health, eq["criticality"], days_since_maint)

            records.append({
                "machine_id":               machine_id,
                "machine_name":             eq["name"],
                "category":                 eq["category"],
                "criticality":              eq["criticality"],
                "power_kw":                 eq["power_kw"],
                "design_life_days":         eq["design_life_days"],
                "installation_date":        install_date.strftime("%Y-%m-%d"),
                "observation_date":         obs_date.strftime("%Y-%m-%d"),
                "machine_age_days":         age_days + obs * 15,
                "avg_ops_hours_per_day":    avg_ops_hours,
                "total_runtime_hours":      total_runtime + obs * avg_ops_hours * 15,
                "avg_temperature_c":        temp,
                "avg_vibration_mmps":       vibration,
                "avg_current_ratio":        current,
                "breakdown_count":          breakdowns,
                "days_since_last_maint":    days_since_maint,
                "last_maintenance_date":    last_maint_date.strftime("%Y-%m-%d"),
                "maintenance_cost_inr":     maint_cost_history,
                "health_score":             health,
                "rul_days":                 rul,
                "maintenance_priority":     priority,
            })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_dataset(n_machines=500, obs_per_machine=10)
    out_path = os.path.join(os.path.dirname(__file__), "maintenance_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset saved: {out_path}")
    print(f"Shape: {df.shape}")
    print(df.head(3).to_string())
    print("\nPriority distribution:")
    print(df["maintenance_priority"].value_counts())
    print("\nHealth score statistics:")
    print(df["health_score"].describe())
