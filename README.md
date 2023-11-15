# EX NO : 10   Mini Project 
## Rainfall Analysis of India

## DATE :

## Description :
As is widely recognized, rainfall data is necessary for the mathematical modelling of extreme hydrological events, such as droughts or floods, as well as for evaluating surface and subsurface water resources and their quality. The phase, quantity, and elevation of generic hydrometeors in the atmosphere can be estimated by ground-based radars. Satellites can provide images with visible and infrared radiation, and they can also serve as platforms for radiometers to derive the quantity and phase of hydrometeors. Radars and satellites provide spatial information on precipitation at wide scales, avoiding many problems connected to local ground measurements, including those for the areal inhomogeneity of a network. However, direct rainfall observations at point scale can be obtained only by rain gauges installed at the soil surface.

## KEY FEATURES :
Three main characteristics of rainfall are its amount, frequency and intensity, the values of which vary from place to place, day to day, month to month and also year to year. Precise knowledge of these three main characteristics is essential for planning its full utilization.

## CODE :
DEVELOPED BY : SYED JAVEED H

REGISTER NO : 212221220055


```
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pickle
from sklearn.model_selection import train_test_split
#read the whole file 
df.iloc[:]
df.info()
print(f'Number of Rows: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')
df.columns
row3 = df.iloc[3]
print(row3)
print("\n")
row115 = df.iloc[115]
print (row115)
description = df.describe()
print(description)
#plot the actual rain 
dg = df[['jun','jul','aug','sep']]
dg.plot(figsize=(20, 5));
#plot shows the total rain 
dh = df[['total']]
dh.plot(figsize=(10, 10));
#dy represent dataframe for 10 last rows which is rainfall from year 2007 to 2016 
dy = df.tail(10)
print(dy)
#plot diagram shows only "TOTAL" rain from dy dataframe (rainfall from year 2007 to 2016)
de = dy[['total']]
de.plot(figsize=(10, 10));
#rainfall from 2007 to 2016
#plot actual rain and departure percentage 
a_d = dy[['jun', 'jul', 'aug', 'sep', 'jun_p', 'jul_p', 'aug_p', 'sep_p']]
a_d.plot(figsize=(20, 10));
#rainfall from 2007 to 2016
#only plot actual rain 
actual = dy[['jun', 'jul', 'aug', 'sep']]
actual
actual = dy[['jun', 'jul', 'aug', 'sep']]
# Calculate the rainfall ranges for each month
rainfall_ranges = actual.max() - actual.min()
# Display the rainfall ranges, minimum, and maximum values
for month, rainfall_range in rainfall_ranges.items():
    min_rainfall = actual[month].min()
    max_rainfall = actual[month].max()
    
    print(f"Rainfall Range for {month}: Range = {rainfall_range} mm,", end=' ')
    print(f"Min = {min_rainfall} mm, Max = {max_rainfall} mm")
actual.plot(figsize=(20, 10))
# Add a title and labels for the axes
plt.title('Actual Rainfall (mm) vs Year')
plt.xlabel('Year')
plt.ylabel('Actual Rainfall (mm)')
plt.show()
#rainfall from 2007 to 2016
#plot departure percentage 
de = dy[['jun_p', 'jul_p', 'aug_p', 'sep_p']]
de
de = dy[['jun_p', 'jul_p', 'aug_p', 'sep_p']]
# Calculate the rainfall ranges for each month
rainfall_ranges = de.max() - de.min()
# Display the rainfall ranges, minimum, and maximum values
for month, rainfall_range in rainfall_ranges.items():
    min_rainfall = de[month].min()
    max_rainfall = de[month].max()
    
    print(f"Rainfall Range for {month}: Range = {rainfall_range} %,", end=' ')
    print(f"Min = {min_rainfall} %, Max = {max_rainfall} %")
print("\n")
de.plot(figsize=(20, 10))
# Add a title and labels for the axes
plt.title('Departure Percentage vs Year')
plt.xlabel('Month')
plt.ylabel('Departure Percentage (%)')
plt.show()
#From 1901 to 2016
actual_df = df[['jun', 'jul', 'aug', 'sep']]
actual_df
# Calculate the average rainfall for each month
av_df = actual_df.mean()
av_df
#Stacked area chart
#Calculate the cumulative rainfall for each month or season
cumulative_rainfall = df[['jun', 'jul', 'aug', 'sep']].cumsum(axis=1)
# Plot the stacked area chart
plt.figure(figsize=(10, 6))
plt.stackplot(df.index, cumulative_rainfall.values.T, labels=['Jun', 'Jul', 'Aug', 'Sep'])
plt.title('Cumulative Rainfall Over Time')
plt.xlabel('Year')
plt.ylabel('Cumulative Rainfall (mm)')
plt.legend(loc='upper left')
plt.show()
#Scatter plot
# Create a scatter plot with trendline
plt.figure(figsize=(10, 6))
sns.regplot(x=actual_df.index, y='jun', data=actual_df, scatter=True, label='June')
sns.regplot(x=actual_df.index, y='jul', data=actual_df, scatter=True, label='July')
sns.regplot(x=actual_df.index, y='aug', data=actual_df, scatter=True, label='August')
sns.regplot(x=actual_df.index, y='sep', data=actual_df, scatter=True, label='September')
plt.title('Rainfall Variation over Time')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.show()
# Reset display options to default
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
actual_df
# Plotting the predicted values versus the actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()
```
## OUPUT:
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/e949b4308a72962fe55aa2e5b668981279d016f7/output1.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%202.png)

![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output3.png)
print(f'Number of Rows: {df.shape[0]}') : to print the number of rows
print(f'Number of Columns: {df.shape[1]}') : to print number of columns
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%204.png)
df.columns is to retrieve the column labels or column names of a DataFrame
to shows data at row 3 (year 1904) and row 115 (year 2016)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%205.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%206.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%206.1.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%207.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%208.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%209.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2010.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2011.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output12.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2013.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2014.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2015.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2016.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2017.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2018.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2019.png)
Based on the cumulative rainfall data for the months of June, July, August, and September from 1901 to 2016,
The cumulative rainfall values represent the total amount of rainfall received from June to September for each year. 
The data shows variations in cumulative rainfall over time, indicating fluctuations in precipitation patterns.

![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2021.png)
![image](https://github.com/sathiya7g/Data-science-mini-project/blob/main/output%2022.png)


## RESULT:
Thus the rainfall analysis has been executed successfully.
