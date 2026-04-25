# VCR-Utilization-Prediction
Predict Utilization of Cisco VCR Devices using XGBoost

## Project Overview
This project uses **XGBoost Machine Learning** to predict the weekly utilization of 1,000+ VCR devices. 

## Key Features
* **Automated Data Ingestion:** Stacks 28 weeks of telemetry reports.
* **PII Masking:** Custom character mapping to anonymize device names.
* **Time-Series Engineering:** Uses 4-week rolling averages and lag features.
* **Forecasting:** Recursive logic to predict usage 4 weeks into the future.

## Results
* **Mean Absolute Error (MAE):** 4.0 Hours
* **Top Predictor:** 4-Week Rolling Average
* **Accuracy:** ~90% for typical usage ranges.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python src/main.py`
