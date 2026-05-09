#Visualizations

import matplotlib.pyplot as plt
import seaborn as sns
import os

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

#Folder Creation for Plots
if not os.path.exists('Report_Plots'):
    os.makedirs('Report_Plots')

######################--------Visualization-------####################################################################

def Visualization_Plot(Master_DFRDD):

    #Plot Data for Visualization for Overall Trend #Average Utilization per week across the whole devices
    Utilization_Trend = Master_DFRDD.groupby('ReportStart')['Total Hours Used'].mean().reset_index()

    plt.figure(figsize=(12,6))
    sns.set_style("whitegrid")
    sns.lineplot(data= Utilization_Trend, x='ReportStart', y= 'Total Hours Used', marker='o', color = 'royalblue', linewidth='2.5')
    plt.title('Average Device Utilization of Past Weeks', fontsize=15)
    plt.ylabel('Average Hours Used')
    plt.xlabel('Week Start Date')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha =0.6)
    plt.savefig('Report_Plots/Utilization_Trend.png', dpi=300, bbox_inches='tight')
    #plt.show()
    print("Saved: Utilization_Trend.png")

#####################--------Visualization-------####################################################################

#Plot Data for Visualization Sample Devices

    #Random Device Selection
    Random_Device = Master_DFRDD['Assigned To'].drop_duplicates().sample(5).values
    RDD_Sample = Master_DFRDD[Master_DFRDD['Assigned To'].isin(Random_Device)]

    plt.figure(figsize=(12,6))
    sns.lineplot(data= RDD_Sample, x='ReportStart', y= 'Total Hours Used', hue = 'Assigned To')
    plt.title('Utilization Pattern of Random Devices')
    plt.legend(loc = 'upper left', bbox_to_anchor = (1.05,1))
    plt.xticks(rotation=45)
    plt.savefig('Report_Plots/Random_Device_plot.png', dpi=300, bbox_inches='tight')
    #plt.show()
    print("Saved: Random_Device_plot.png") 

#####################--------Visualization-------####################################################################

    #Plot Data for Corelation Visualization via Heatmap
    RDD_Numerical = Master_DFRDD[['Total Hours Used', 'Calls','Local Display Wired', 'USB Passthrough', 'Target']]

    plt.figure(figsize=(10,8))
    RDD_Corr = RDD_Numerical.corr()
    sns.heatmap(RDD_Corr, annot=True, cmap = 'coolwarm', fmt = '.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('Report_Plots/Heatmap_plot.png', dpi=300, bbox_inches='tight')
    #plt.show()
    print("Saved: Heatmap_plot.png")

    #Plot Data for Usage Visualization via Heatmap
    
    Top_Devices = Master_DFRDD.groupby('Assigned To')['Total Hours Used'].sum().nlargest(25).index
    TD_Subset = Master_DFRDD[Master_DFRDD['Assigned To'].isin(Top_Devices)] 

    Pivot_DFRDD = TD_Subset.pivot_table(index='Assigned To', columns = 'ReportStart', values = 'Total Hours Used')

    plt.figure(figsize=(14,8))
    sns.heatmap(Pivot_DFRDD, cmap="YlGnBu", annot=False, cbar_kws={'label':'Hours Used'})
    plt.title('Heatmap: Top 25 Most Active Devices (Usage Intensity)', fontsize=15)
    plt.xlabel('Report Date')
    plt.xticks(ticks=[i + 0.5 for i in range(len(Pivot_DFRDD.columns))],labels=Pivot_DFRDD.columns, rotation =45, ha = 'right')
    plt.ylabel('Device (Assigned To)')
    plt.savefig('Report_Plots/Heatmap Top 25 Most Active Devices.png', dpi=300, bbox_inches='tight')
    #plt.show()
    print("Saved: Heatmap: Top 25 Most Active Devices.png")

#####################--------Visualization-------####################################################################

    #Boxplot Comparison of Total Hours Used
    plt.figure(figsize=(10,5))
    sns.set_style("whitegrid")
    sns.boxplot(x=Master_DFRDD['Total Hours Used'], color='skyblue', showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black","markersize":"10"})
    plt.title('Boxplot for Total Hours Used', fontsize=15)
    plt.xlabel('Weekly Hours Used')
    plt.savefig('Report_Plots/Boxplot for Total Hours Used.png', dpi=300, bbox_inches='tight')
    #plt.show()
    print("Saved: Boxplots for Total Hours Used.png")

    #Outlier Detetction
    Q1 = Master_DFRDD['Total Hours Used'].quantile(0.25)
    Q3 = Master_DFRDD['Total Hours Used'].quantile(0.75)
    IQR = Q3 - Q1

    upper_bound = Q3 + 1.5 * IQR
    outliers = Master_DFRDD[Master_DFRDD['Total Hours Used'] > upper_bound]
    outlier_count = len(outliers)
    unique_outlier_devices = outliers['Assigned To'].nunique()

    print(f"Total Outlier Weeks: {outlier_count}")
    print(f"Number of Unique Devices showing outlier behavior: {unique_outlier_devices}")
    print(f"The 'Right Whisker' ends at: {upper_bound:.2f} hours")
    print(outliers[['Assigned To', 'ReportStart', 'Total Hours Used']].sort_values(by='Total Hours Used', ascending=False).head(10))

    #####################--------Visualization-------####################################################################

    #Visualize the Holiday Effect
    plt.figure(figsize=(12,6))
    sns.lineplot(data = Master_DFRDD, x = 'ReportStart', y = 'Total Hours Used', estimator='mean')
    plt.axvspan('2025-12-25','2025-12-31', color ='red', alpha = 0.1, label = 'Holiday Weeks' )
    plt.title('Utilization Holiday Imapct')
    plt.legend()
    plt.savefig('Report_Plots/Utilization Holiday Imapct.png', dpi=300, bbox_inches='tight')
    #plt.show()
    print("Saved: Utilization Holiday Imapct.png")

#####################--------Visualization-------####################################################################
