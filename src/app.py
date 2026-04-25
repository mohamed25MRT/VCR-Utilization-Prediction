#Dashboard Application for Easy View

#Import Libraries
import streamlit as st
import pandas as pd
import plotly.express as px

from Data_Processor import data_extract
from Model_Train import Train_Model
from Forecasts_Prediction import Week_prediction

st.set_page_config(page_title="VCR Utilization AI", layout="wide")

# Cache the data so the app stays fast
@st.cache_data
def get_data_and_forecast():
    Master_DFRDD = data_extract("C:/Users/DELL/projects/CS5998/RDD/")
    model, model_features = Train_Model(Master_DFRDD)
    forecast_results = Week_prediction(model, Master_DFRDD, model_features)
    return Master_DFRDD, forecast_results

Master_DFRDD, forecast_results = get_data_and_forecast()

st.title ("Cisco VCR Utilization Forecast")

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