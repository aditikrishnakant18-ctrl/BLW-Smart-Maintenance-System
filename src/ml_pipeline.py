import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import joblib

from src.database import fetch_all_equipment
from src.health_engine import calculate_health_score

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def prepare_targets(df: pd.DataFrame):
    df['health_score'] = df.apply(calculate_health_score, axis=1)
    
    base_rul = (df['health_score'] / 100.0) * 120.0
    noise = np.random.normal(0, 2, len(df))
    df['RUL'] = np.clip(base_rul + noise, 1, 120).astype(int)
    
    def map_priority(hs):
        if hs < 50: return "Critical"
        elif hs < 70: return "High"
        elif hs < 85: return "Medium"
        else: return "Low"
        
    df['priority'] = df['health_score'].apply(map_priority)
    return df

def train_pipeline():
    df = fetch_all_equipment()
    if df.empty:
        raise ValueError("Database is empty. Populate database using generator.py first.")
        
    df = prepare_targets(df)
    
    features = [
        "machine_age_days", "total_runtime_hours", "breakdown_count", 
        "days_since_last_maint", "avg_temp_c", "avg_vibration_mms", "avg_current_amp"
    ]
    
    X = df[features]
    y_rul = df['RUL']
    y_priority = df['priority']
    
    X_train, X_test, y_rul_train, y_rul_test, y_pri_train, y_pri_test = train_test_split(
        X, y_rul, y_priority, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training RUL Regressor...")
    rul_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rul_model.fit(X_train_scaled, y_rul_train)
    rul_preds = rul_model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_rul_test, rul_preds)
    r2 = r2_score(y_rul_test, rul_preds)
    print(f"RUL Model Complete. MAE: {mae:.2f} Days, R2: {r2:.4f}")
    
    print("Training Priority Classifier...")
    pri_model = RandomForestClassifier(n_estimators=100, random_state=42)
    pri_model.fit(X_train_scaled, y_pri_train)
    pri_preds = pri_model.predict(X_test_scaled)
    
    print("Priority Model Classification Report:")
    print(classification_report(y_pri_test, pri_preds))
    
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(rul_model, os.path.join(MODELS_DIR, "rul_regressor.joblib"))
    joblib.dump(pri_model, os.path.join(MODELS_DIR, "priority_clf.joblib"))
    print("Model training completed and artifacts saved.")

if __name__ == "__main__":
    train_pipeline()