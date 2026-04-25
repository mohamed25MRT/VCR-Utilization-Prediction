#Function for Predicting Utilization 
#Use the Trained Model and Predict Utilization for Upcoming Weeks.

def Week_prediction(model, Master_DFRDD, Model_Features):
    #Future Weeks Forecasting

    Last_Week = Master_DFRDD[Master_DFRDD['ReportStart'] == Master_DFRDD['ReportStart'].max()].copy()
    Forecasts = []

    Present_Data = Last_Week.copy()

    for i in range (1,5):
        #Next_Week Prediction
        X_Present_Week = Present_Data[Model_Features].copy()
        Predictions = model.predict(X_Present_Week)

        #Storing Prediction
        Week_No = f"Week_+{i}"
        Present_Data[Week_No] = Predictions
        Forecasts.append(Present_Data[['Assigned To', Week_No]])

        #Update Lags for the Next Iteration
        Present_Data['Total Hours Used Lag1'] = Present_Data['Total Hours Used']
        Present_Data['Total Hours Used'] = Predictions

        Present_Data['Rolling_AVG_4WK'] = (Present_Data['Rolling_AVG_4WK'] * 3 + Predictions) / 4


    #Merge All Forecasts
    Final_Forecasts = Forecasts[0]
    for F in Forecasts[1:]:
        Final_Forecasts = Final_Forecasts.merge(F, on='Assigned To')

    print("Forecast Predictions")
    print(Final_Forecasts.head())

    return Final_Forecasts