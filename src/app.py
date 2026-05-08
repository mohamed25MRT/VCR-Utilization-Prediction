#Dashboard Application for Easy View

#Import Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from Data_Processor import data_extract
from Model_Train import Train_Model
from Forecasts_Prediction import Week_prediction
from Data_Processor_RNN import data_extract_RNN
from Seq2Seq_Pred import Seq_2_Seq_Model,Seq2seq_Data
from mainRNN import mainRNN

st.set_page_config(page_title="VCR Utilization AI", layout="wide")

# Cache the data so the app stays fast
@st.cache_data
def get_data_and_forecast():
    #A. XGBoost Logic
    Master_DFRDD = data_extract("C:/Users/DELL/projects/CS5998/RDD")
    model, model_features = Train_Model(Master_DFRDD)
    forecast_results = Week_prediction(model, Master_DFRDD, model_features)

    rnn_forecast = pd.read_csv("rnn_fleet_forecast.csv")

    return Master_DFRDD, forecast_results, rnn_forecast

Master_DFRDD, forecast_results, rnn_forecast = get_data_and_forecast()

st.title ("Cisco VCR Utilization Forecast")

# Assuming your variables are rnn_mae = 4.2 and xgb_mae = 4.0
st.header("🏆 Model Performance Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="XGBoost MAE", 
        value=f"{4.00:.2f} hrs", 
        help="Average error per week for the Tabular model."
    )

with col2:
    # This shows the RNN MAE and highlights that it is 0.2 higher than XGBoost
    st.metric(
        label="RNN Seq2Seq MAE", 
        value=f"{4.20:.2f} hrs", 
        delta="-0.20", 
        delta_color="inverse",
        help="Average error across the 4-week forecast horizon."
    )

with col3:
    st.metric(
        label="Target Accuracy", 
        value="< 5.00 hrs", 
        delta="Goal Met", 
        delta_color="normal"
    )

st.divider()

#Download Option
st.sidebar.header("Data Actions")
csv = forecast_results.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download 4-Week Forecast CSV",
    data=csv,
    file_name='vcr_forecast_report.csv',
    mime='text/csv',
)

# Dashboard UI
Device_list = [""] + list(Master_DFRDD['Assigned To'].unique())
st.sidebar.header("Controls")
Selected_Device = st.sidebar.selectbox(
    "Select a Device ID to begin:", 
    options=Device_list,
    index=0  # This ensures it starts on the empty string ""
)

Prediction_DF = forecast_results[forecast_results['Assigned To'] == Selected_Device]

#Plotly Chart
#Figure = px.line(Device_DF, x = 'ReportStart', y ='Total Hours Used', title = f"Trend for {Selected_Device}")
#st.plotly_chart(Figure)

st.write("4 Week Prediction")
st.dataframe(Prediction_DF)

if Selected_Device == "":
    
    st.info("Please select a Device ID from the sidebar to view utilization and forecasts.")

    st.metric("Total Devices Monitored", len(Master_DFRDD['Assigned To'].unique()))
    st.metric("Latest Report Date", Master_DFRDD['ReportStart'].max().strftime('%Y-%m-%d'))

else:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Historical Usage")
        hist_data = Master_DFRDD[Master_DFRDD['Assigned To'] == Selected_Device]
        st.line_chart(hist_data.set_index('ReportStart')['Total Hours Used'])

    with col2:
        st.subheader("Upcoming 4-Week Prediction")
        dev_forecast = forecast_results[forecast_results['Assigned To'] == Selected_Device].iloc[:,1:].T
        #st.dataframe(dev_forecast.iloc[1:], columns=["Predicted Hours"])
        #st.dataframe(dev_forecast.iloc[1:])
        dev_forecast.columns = ["predicted Hours"]
        st.table(dev_forecast)

        next_week = dev_forecast.iloc[0, 0]
        if next_week > 40:
            st.warning(f"High usage predicted: {next_week:.1f} hrs")


# Load the pre-calculated RNN results
    st.divider()

    # --- RNN Section ---
    st.subheader(f"RNN Seq2Seq Prediction for {Selected_Device}")
    
    # Filter the RNN dataframe for the selected device
    dev_rnn_row = rnn_forecast[rnn_forecast['Assigned To'] == Selected_Device]

    if not dev_rnn_row.empty:
        # Transpose it so it looks like a vertical table (matching XGBoost style)
        dev_rnn_display = dev_rnn_row.drop(columns=['Assigned To']).T
        dev_rnn_display.columns = ["RNN Predicted Hours"]
        
        # Display the table and a small plot
        col_table, col_plot = st.columns([1, 2])
        
        with col_table:
            st.table(dev_rnn_display)
            
        with col_plot:
            # Simple line chart for the 4-week trend
            st.line_chart(dev_rnn_display)
    else:
        st.warning("No RNN sequence data available for this device (requires 8 weeks of history).")