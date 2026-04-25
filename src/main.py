#Main Python File

#Import Libraries
import pandas as pd

#Function Calls
from Data_Processor import data_extract
from Model_Train import Train_Model
from Forecasts_Prediction import Week_prediction
from Visualizer import Visualization_Plot
#FolderPath = "C:/Users/DELL/projects/CS5998/RDD" 

def VCR_Util_pipeline():
    print("---Starting Cisco VCR Utilization Pipeline ---")
    
    # 1. Import, Extract and Perform Feature Modeling (Create The Master Data Frame)
    print("### Step 1: Loading and processing weekly Utilization data...")
    Master_DFRDD = data_extract("C:/Users/DELL/projects/CS5998/RDD")
    print(f"Loaded {len(Master_DFRDD)} rows across {Master_DFRDD['Assigned To'].nunique()} devices.")
    print("Master_DFRDD Columns : ", Master_DFRDD.columns.tolist())
    
    # 2. Train Model
    print("\n### Step 2: Training XGBoost Model...")
    model, model_features = Train_Model(Master_DFRDD)
    print("Model training complete and saved as 'vcr_model.json'.")
    
    
    # 3. Generate 4-Week Forecast
    print("\n### Step 3: Generating 4-week recursive forecast...")
    forecast_results = Week_prediction(model, Master_DFRDD, model_features)
    
    # 4. Export
    output_file = "Final_VCR_Forecast_Report.csv"
    forecast_results.to_csv(output_file, index=False)

    # 5. Visualization
    Visualization_Plot(Master_DFRDD)
    
    print(f"\n--- ✅ Success! ---")
    print(f"Final forecast for 1,000 devices exported to: {output_file}")
    
    # Quick summary for the console
    avg_util = forecast_results[['Week_+1', 'Week_+2', 'Week_+3', 'Week_+4']].mean()
    print("\nFleet-Wide Predicted Averages:")
    print(avg_util)

if __name__ == "__main__":
    VCR_Util_pipeline()