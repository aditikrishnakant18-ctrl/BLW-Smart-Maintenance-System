# BLW Smart Maintenance System

## AI-Based Predictive Maintenance & Asset Health Monitoring System

A machine-learning-enabled predictive maintenance platform designed to monitor industrial equipment, assess asset health, identify abnormal operating conditions, and support maintenance planning.

## Project Overview

The system uses historical equipment and maintenance data along with live machine parameters to estimate equipment health and maintenance priority.

It provides an interactive Streamlit dashboard where users can:

- Select equipment by category and ID
- View machine operational parameters
- Simulate machine conditions and failure scenarios
- Calculate an equipment health score
- Determine maintenance priority
- Estimate remaining useful life
- Generate a recommended maintenance date
- View anomaly explanations
- Identify important prediction features
- Receive engineering-oriented maintenance recommendations

## Key Features

### Asset Diagnostics
View equipment-specific operational information and maintenance history.

### Live Machine Simulation
Modify parameters such as:

- Temperature
- Vibration
- Current
- Days since last maintenance

to simulate changing machine conditions.

### Failure Simulation
Simulate conditions such as:

- Bearing failure
- Cooling failure
- Lubrication failure

and observe their effect on the equipment condition.

### Health & Maintenance Prediction

The system provides:

- Health Score
- Maintenance Priority
- Remaining Useful Life
- Recommended Maintenance Date

### Anomaly Explanation

The dashboard highlights abnormal conditions and shows important prediction drivers using feature weights.

### Engineering Recommendations

The system generates maintenance recommendations based on the detected machine condition.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- SQLite
- Matplotlib

## How to Run

Clone the repository:

git clone https://github.com/aditikrishnakant18-ctrl/BLW-Smart-Maintenance-System.git

Navigate into the project:
cd BLW-Smart-Maintenance-System
Install dependencies:
pip install -r requirements.txt
Run the Streamlit application:
streamlit run app.py

## Future Improvements

* Integration with real-time IoT sensor data
* Improved predictive models using larger industrial datasets
* Automated maintenance alerts
* Model monitoring and retraining
* Integration with enterprise maintenance management systems
