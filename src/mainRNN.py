import pandas as pd
import tensorflow as TF
import numpy as np
import matplotlib.pyplot as plt

from Seq2Seq_Pred import Seq2seq_Data, Seq_2_Seq_Model, InverseScale_output
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from Data_Processor_RNN import data_extract_RNN

def mainRNN():

    #Define Base Parameters
    PastWeeks = 8
    FutureWeeks = 4
    RNN_Fetaures = 2
    Data_Path = "C:/Users/DELL/projects/CS5998/RDD"

    #df = pd.read_csv("master_vcr_data.csv") # Your vertically stacked data
    print("---Starting Cisco VCR Utilization RNN/Seq2Seq Pipeline ---")
    
    # 1. Import, Extract and Perform Feature Modeling (Create The Master Data Frame)
    print("### Step 1: Loading and processing weekly Utilization data...")
    Master_DFRDD_RNN = data_extract_RNN(Data_Path)
  
    print(f"Loaded {len(Master_DFRDD_RNN)} rows across {Master_DFRDD_RNN['Assigned To'].nunique()} devices.")
    print("Master_DFRDD Columns : ", Master_DFRDD_RNN.columns.tolist())
       
        
    # 2. Scaling for Deep Learning
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = Master_DFRDD_RNN.copy()
    df_scaled[['Total Hours Used', 'Is_Holiday']] = scaler.fit_transform(Master_DFRDD_RNN[['Total Hours Used', 'Is_Holiday']])

    # 3. Create Sequences (8 weeks past -> 4 weeks future)
    print("--- Generating 3D Tensors for Seq2Seq ---")
    
    X, y = Seq2seq_Data(df_scaled, 8, 4)
    #y = np.expand_dims(y, axis=-1)
    print(f"DEBUG: X shape is {X.shape}")
    print(f"DEBUG: y shape is {y.shape}")

    # Split into Train and Test (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 4. Build and Train Model
    print("--- Initializing Encoder-Decoder LSTM ---")
    modelRNN = Seq_2_Seq_Model(PastWeeks, RNN_Fetaures, FutureWeeks)
    
    history = modelRNN.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)
    
        
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train MAE (Loss)')
    plt.plot(history.history['val_loss'], label='Validation MAE')
    plt.title('RNN Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Error (Hours)')
    plt.legend()
    plt.savefig('Report_Plots/RNN Learning Curve.png', dpi=300, bbox_inches='tight')
    #plt.show()
    print("Saved: RNN Learning Curve.png")

    # --- 5. Evaluation ---
    print("Step 4: Evaluating on Unseen Test Data...")
    raw_predictions = modelRNN.predict(X_test)

    # 6. Inverse Scale back to Hours
    final_preds_2d = raw_predictions.reshape(raw_predictions.shape[0], raw_predictions.shape[1])
    final_actuals_2d = y_test.reshape(y_test.shape[0], y_test.shape[1])

    final_preds = InverseScale_output(final_preds_2d, scaler)
    final_actuals = InverseScale_output(final_actuals_2d, scaler)

    # 7. Calculate MAE
    mae_score = mean_absolute_error(final_actuals, final_preds)
    print(f"\n--- FINAL RESULTS ---")
    print(f"Seq2Seq Mean Absolute Error: {mae_score:.2f} hours")

    # 8. Generate Forecast
    print("--- Generating Forecast for Dashboard ---")
    latest_windows = []
    device_ids = []

    for device, group in Master_DFRDD_RNN.groupby('Assigned To'):
        if len(group) >= PastWeeks:
            # Grab the last 8 weeks and ensure columns match training
            window = group.sort_values('WeekNumber')[['Total Hours Used', 'Is_Holiday']].values[-PastWeeks:]
            # Apply scaling
            scaled_window = scaler.transform(window)
            latest_windows.append(scaled_window)
            device_ids.append(device)

    X_fleet = np.array(latest_windows) # Shape: (Num_Devices, 8, 2)

    raw_preds_fleet = modelRNN.predict(X_fleet) # Shape: (Num_Devices, 4, 1)

    preds_2d = raw_preds_fleet.reshape(raw_preds_fleet.shape[0], FutureWeeks)

    h_max = Master_DFRDD_RNN['Total Hours Used'].max()
    h_min = Master_DFRDD_RNN['Total Hours Used'].min()
    final_forecast_hours = preds_2d * (h_max - h_min) + h_min

    rnn_forecast_results = pd.DataFrame(final_forecast_hours, columns=['Week+1', 'Week+2', 'Week+3', 'Week+4'])
    rnn_forecast_results.insert(0, 'Assigned To', device_ids)

    rnn_forecast_results.to_csv("rnn_fleet_forecast.csv", index=False)
    print(f"Success: Forecast generated for {len(device_ids)} devices.")
    

    # 9. Save model for Streamlit
    modelRNN.save("vcr_seq2seq_v1.h5")
    print("Project Pipeline Complete: Seq2Seq Model Saved.")

    return modelRNN, mae_score, scaler, final_actuals, final_preds

if __name__ == "__main__":
    mainRNN()