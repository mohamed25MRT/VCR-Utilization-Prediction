# VCR-Utilization-Prediction/Forecasting Engine
Predict Utilization of Cisco VCR Devices using XGBoost and a Seq2Seq Model. This project implements a dual-model approach to balance short-term precision with long-term trend visibility.

## Project Overview
This project uses **XGBoost Machine Learning** to predict the weekly utilization of 1,000+ VCR devices. It also helps capacity planning by providing:
- **Tactical Alerts:** 1-week predictions via XGBoost.
- **Strategic Trends:** 4-week forecast horizons via a Seq2Seq RNN

## Key Features
* **Automated Data Ingestion:** Stacks 40 weeks of telemetry reports.
* **PII Masking:** Custom character mapping to anonymize device names (Not Implemented Due to Complexity).
* **Time-Series Engineering:** Uses 4-week rolling averages and lag features.
* **Forecasting:** Recursive logic to predict usage 4 weeks into the future.

## Modeling Approaches
### 1. XGBoost (Baseline)
- **Architecture:** Gradient Boosted Decision Trees.
- **Input:** 2-week lagged usage + Holiday flags.
- **Performance:** **4.00 MAE**.
- **Best For:** Immediate, high-precision weekly capacity balancing.

### 2. RNN Seq2Seq (Advanced)
- **Architecture:** Encoder-Decoder LSTM (Long Short-Term Memory).
- **Input:** 8-week sequential "lookback" windows.
- **Output:** 4-week continuous forecast horizon.
- **Performance:** **4.20 MAE**.
- **Best For:** Identifying long-term growth momentum and month-ahead planning.

## Results
* **Mean Absolute Error (MAE):** 4.0 Hours (XGBoost) and 4.2 Hours (RNN Seq2Seq)
* **Top Predictor:** 4-Week Rolling Average
* **Accuracy:** ~90% for typical usage ranges.

## Installation & Usage

## How to Run
1.**Clone the repository:**
   git clone [https://github.com/mohamed25MRT/VCR-Utilization-Prediction.git](https://github.com/mohamed25MRT/VCR-Utilization-Prediction)
   cd VCR-Utilization-Prediction
   
2. Install dependencies: pip install -r requirements.txt
   
3. Dashboard: streamlit run app.py

**## How to RunDashboard Features** 
**Device Selection**: Interactive lookup for any device in the 1,000+ fleet.
**Model Comparison**: Side-by-side visualization of XGBoost vs. RNN predictions.
**Global Metrics**: Real-time tracking of MAE and RMSE across the entire fleet

**Project Structure**
Data_Processor.py: Ingestion and cleaning of 40+ raw CSV logs.
main.py: Training and evaluation of the XGBoost baseline.
mainRNN.py: Implementation of the Encoder-Decoder LSTM architecture.
app.py: Streamlit-based visual reporting dashboard.
Other files..
