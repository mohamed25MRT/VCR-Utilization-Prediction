#Function for Training the XGBoost Model 
#Use the converted data and traing the XGBoost Model with Hyperparameters

#Import Libraries
import xgboost as xgb
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def Train_Model(Master_DFRDD):
    #The XGB Model for Prediction

    #Drop Any Rows Where The Target is Zero [XGBoost Cannot Handle NAN or Large Values Usually the Last Week Data on the Master Dataset]
    #Master_DFRDD = Master_DFRDD.dropna(subset=['Target', 'Total Hours Used_lag1'])
    Master_DFRDD = Master_DFRDD.dropna(subset=['Target'])

    #Drop Non Numeric Values from Training Data in Input - "X"
    X = Master_DFRDD.drop(['Target', 'ReportStart','ReportEnd', 'Device Type', 'Assigned To'], axis=1)
    y = Master_DFRDD['Target']

    #Split Data for Training and Testing based on Time
    PastWeek = Master_DFRDD['ReportStart'].max()
    Train = Master_DFRDD['ReportStart'] < PastWeek
    Test = Master_DFRDD['ReportStart'] == PastWeek

    Train_Size = int(len(X) *0.8)

    X_Train, X_Test = X.iloc[:Train_Size], X.iloc[Train_Size:]
    y_Train, y_Test = y.iloc[:Train_Size], y.iloc[Train_Size:]

    print("X_Train NAN Count : ",X_Train.isnull().sum())
    print("y_Train NAN Count : ",y_Train.isnull().sum())
        
    X_Train = X_Train.fillna(0)
    X_Test = X_Test.fillna(0)

    #InitializeAndTrain
    model = xgb.XGBRegressor(
        n_estimators =500,
        learning_rate=0.05,
        max_depth=6,
        subsample =0.8,
        random_state = 42,
        tree_method='hist',
        colsample_bytree = 0.8,
        enable_categorical = True
    )

    #Fit the Model
    #Input X --> Contains the Usage utilization such as Hours of Calls and Local Display from previous weeks
    #Output y --> Contains the output utilization that is being predicted
    model.fit(
        X_Train, y_Train, 
        eval_set=[(X_Test, y_Test)], 
        verbose =100
    )

    #Make Predictions
    Predictions = model.predict(X_Test)

    #Evaluating the Results : Calculating Accuracy and Feature Importance

    MAE_XGB = mean_absolute_error(y_Test, Predictions)
    print(f"Mean Absolute Error : {MAE_XGB: .2f} Hours")

    Feature_Importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    print(Feature_Importance.sort_values(by='Importance', ascending=False))

    #Error
    Residual_XGB = y_Test -Predictions

######################--------Visualization-------####################################################################
    #Visualize Predicted versus Actual 

    plt.figure(figsize=(12,6))
    plt.plot(y_Test.values[:50], label='Actual Usage', color='royalblue', linewidth=2)
    plt.plot(Predictions[:50], label='XGBoost Prediction', color='tab:red', linestyle='--', alpha=0.8)
    
    plt.title('Actual vs. Predicted Utilization', fontsize=15)
    plt.ylabel('Hours Used')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('Report_Plots/Actual vs. Predicted.png', dpi=300, bbox_inches='tight')
    #plt.show()
    print("Saved: Actual vs. Predicted.png")
######################--------Visualization-------####################################################################
    plt.figure(figsize=(12,6))
    sns.scatterplot(x=Predictions, y=Residual_XGB, color ='purple', alpha = 0.5)
    plt.axhline(y= 0 , color = 'black', linestyle='--')
    plt.title("Error Plot (Residual Value)", fontsize =15)
    plt.xlabel("Predicted Hours")
    plt.ylabel("Error = (Actual - Predicted)")
    plt.savefig('Report_Plots/Error Plot.png', dpi=300, bbox_inches='tight')
    #plt.show()
    print("Saved: Error Plot.png.png")

######################--------Visualization-------####################################################################

    model.save_model("VCR_Model.json")
    return model, X_Train.columns.tolist()


