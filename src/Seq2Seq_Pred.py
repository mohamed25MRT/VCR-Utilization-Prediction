import tensorflow as TF
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector

def Seq2seq_Data(DF, PastWeeks = 8, FutureWeeks = 4):
    X,y = [], []
    for device, group in DF.groupby('Assigned To'):
        # Sort by week to ensure temporal order
        #Device_Data = group.sort_values('WeekNumber')[['Total Hours Used', 'Is_Holiday']].values
        group = group.sort_values('WeekNumber')

        device_data = group[['Total Hours Used','Is_Holiday']].values

        # Slide the window across the 33 weeks
        for i in range(len(device_data) - PastWeeks - FutureWeeks + 1):
            X.append(device_data[i : i + PastWeeks])
            y.append(device_data[i + PastWeeks : i + PastWeeks + FutureWeeks, 0]) # Only predict Call_Hours (index 0)
            
    X = np.array(X)
    y = np.array(y)

    y = np.expand_dims(y, axis = -1)
    return X, y

def Seq_2_Seq_Model(PastWeeks, Features, FutureWeeks):
    TF.keras.backend.clear_session()
    modelRNN= TF.keras.Sequential([

        #Encoder -> Learns the Past Pattern in Usage
        TF.keras.layers.LSTM(128, activation = 'relu', input_shape=(PastWeeks, Features)),

        TF.keras.layers.RepeatVector(FutureWeeks),

        #Decoder -> Generate the next 4 Weeks
        TF.keras.layers.LSTM(128, activation= 'relu', return_sequences=True),

        #Output -> Final Layer to Get the Specific Hour Values
        TF.keras.layers.TimeDistributed(TF.keras.layers.Dense(1))
    ]) 

    modelRNN.compile(optimizer='adam', loss = 'mae')
    return modelRNN


def InverseScale_output(Predictions, Scaler):
    # Flatten preds to match scaler's 2-column expectation
    dummy = np.zeros((Predictions.size, 2)) 
    dummy[:, 0] = Predictions.flatten()
    inv = Scaler.inverse_transform(dummy)[:, 0]
    return inv.reshape(Predictions.shape)
