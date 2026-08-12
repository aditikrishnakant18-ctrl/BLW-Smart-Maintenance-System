import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join("data", "maintenance.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS equipment (
            machine_id TEXT PRIMARY KEY,
            machine_name TEXT,
            machine_category TEXT,
            installation_date TEXT,
            machine_age_days INTEGER,
            power_rating_kw REAL,
            avg_op_hours_per_day REAL,
            total_runtime_hours REAL,
            breakdown_count INTEGER,
            days_since_last_maint INTEGER,
            last_maint_date TEXT,
            criticality_level TEXT,
            avg_temp_c REAL,
            avg_vibration_mms REAL,
            avg_current_amp REAL,
            maint_cost_history REAL
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(df: pd.DataFrame):
    conn = get_connection()
    df.to_sql("equipment", conn, if_exists="replace", index=False)
    conn.close()

def fetch_all_equipment():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM equipment", conn)
    conn.close()
    return df

def fetch_machine_by_id(machine_id: str):
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM equipment WHERE machine_id = ?", conn, params=(machine_id,))
    conn.close()
    return df.iloc[0] if not df.empty else None