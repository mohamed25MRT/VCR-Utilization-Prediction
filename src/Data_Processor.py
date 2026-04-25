#Function for Data Import and Processing
#Import the available CSV Files and Perform Data Feature Extarction and Data Cleaning
import os
#os.system('pip install seaborn')
#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob


#Function to Extract Reports into a Dataframe
def data_extract(FolderPath):
#"""Loads all RDD files and prepares features."""
#GLOB Option (File Explorer Search) to find all CSVs in a Folder and store it in a list
    File_List = glob.glob(os.path.join(FolderPath,"*.csv"))
    #If Folder is Empty or Incorrect throw an Error
    if not File_List:
        raise FileNotFoundError(f"No DFRDD Files in {FolderPath} or Incorrect Folder")
    
    WeeklyUsage = []
    Temp_WeeklyUsage =[] 
    print("Starting to Load Files.......")
    for file in File_List:
        DFRDD = pd.read_csv(file)
        #Columns_Removal = ['Mac Address', 'Device ID', 'Tags', 'IP Address','Latest Known Status','Delete Date']
        Columns_Retain  = ['Assigned To','Device Type','Total Hours Used', 'Calls','Local Display Wired', 'Local Display Wireless', 'Whiteboarding','Digital Signage','USB Passthrough']

        #Extract Date from Filename
        File_Name = os.path.basename(file)
        print("File Name is : ",File_Name)
        Date_FileName = File_Name.split('_')
        Start_Date = Date_FileName[1]
        End_Date = Date_FileName[2].replace('.csv','')

        #Filtering the Columns
        ColumnsXGB = [c for c in Columns_Retain if c in DFRDD.columns]
        Filter_Device = "Cisco Room Navigator"
        DFRDD = DFRDD[DFRDD['Device Type'] != Filter_Device].copy()
        WeeklyUsage = DFRDD[ColumnsXGB].copy()
                 
        #Filtering Rows [Business Hours Per Week is 40 to 60]
        Hours_Count = len(WeeklyUsage)
        WeeklyUsage = WeeklyUsage[(WeeklyUsage['Total Hours Used'] > 0) & (WeeklyUsage['Total Hours Used'] < 60)]

        WeeklyUsage['ReportStart'] = pd.to_datetime(Start_Date)
        WeeklyUsage['ReportEnd'] = pd.to_datetime(End_Date)

        #Feature Engineering (Holiday Flag) #Handling Holiday Data for Last Week of the Year
        Holiday_Dates = ['2025-12-25','2025-12-31', '2026-01-01']

        WeeklyUsage['Is_Holiday'] = WeeklyUsage['ReportStart'].dt.strftime('%y-%m-%d').isin(Holiday_Dates).astype(int)

        Temp_WeeklyUsage.append(WeeklyUsage)

    #Combine all X Weeks Data   #Master_Dataframe 
    Master_DFRDD = pd.concat(Temp_WeeklyUsage,ignore_index=True)
    
    print(Master_DFRDD.head) #-----> Remove Later

    #The Lag Columns Creation
    print("#########################")
    print(".......Lag Creation......")
    print("#########################")
    #Sorting the Data by Date or by Week Number
    Master_DFRDD['WeekNumber'] = Master_DFRDD['ReportStart'].dt.isocalendar().week

    #Creating Lags (Using last x number of data to predict next data)
    Columns_Int  = ['Total Hours Used', 'Calls','Local Display Wired', 'Local Display Wireless', 'Whiteboarding','Digital Signage','USB Passthrough']

    for col in Columns_Int:
        Master_DFRDD[f'{col}_lag1'] = Master_DFRDD.groupby('Assigned To')[col].shift(1)

    #Rolling Week Average for Trend Incorporation
    Master_DFRDD['Rolling_AVG_4WK'] = Master_DFRDD.groupby('Assigned To')['Total Hours Used'].transform(lambda x:x.rolling(4).mean())

    #Prediction Target (Next Weeks Total Hours)
    Master_DFRDD['Target'] = Master_DFRDD.groupby('Assigned To')['Total Hours Used'].shift(-1)
   
    return Master_DFRDD
