import numpy as np

def calculate_health_score(row) -> float:
    vib = row['avg_vibration_mms']
    if vib <= 2.5:
        f_vib = 1.0
    else:
        f_vib = max(0.0, 1.0 - (vib - 2.5) / 5.5)
        
    temp = row['avg_temp_c']
    if temp <= 70.0:
        f_temp = 1.0
    else:
        f_temp = max(0.0, 1.0 - (temp - 70.0) / 40.0)
        
    breakdowns = row['breakdown_count']
    f_fail = np.exp(-0.35 * breakdowns)
    
    days_since = row['days_since_last_maint']
    if days_since <= 90:
        f_delay = 1.0
    else:
        f_delay = max(0.0, 1.0 - (days_since - 90) / 180.0)
        
    runtime = row['total_runtime_hours']
    f_runtime = max(0.0, 1.0 - (runtime / 12000.0))
    
    age_days = row['machine_age_days']
    f_age = max(0.0, 1.0 - (age_days / 5475.0))
    
    w_vib = 0.25
    w_temp = 0.20
    w_fail = 0.15
    w_delay = 0.15
    w_runtime = 0.15
    w_age = 0.10
    
    score = 100.0 * (
        (w_vib * f_vib) + 
        (w_temp * f_temp) + 
        (w_fail * f_fail) + 
        (w_delay * f_delay) + 
        (w_runtime * f_runtime) + 
        (w_age * f_age)
    )
    return round(score, 1)