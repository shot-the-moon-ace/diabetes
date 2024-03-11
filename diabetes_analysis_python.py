# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 09:41:32 2024

@author: kspen
"""

## diabetes analysis ##
import pandas as pd
import datetime as dt
from PyQt5.QtWidgets import QFileDialog, QApplication
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from itertools import product

# Initialize the application
app = QApplication(sys.argv)

# Open the file dialog to select a file
file_path, _ = QFileDialog.getOpenFileName()

# Use the selected file path to read the file into a DataFrame
if file_path:  # Making sure a file path was selected
    diabetes = pd.read_csv(file_path)
    print(diabetes)

# check the column names, number of columns, null and non null values in columns
# check the dtypes of the columns, also making a copy of the original dataframe
# in the event we want to be able to look at the changes made to the dataframe
# over time
diabetes.info()
diabetes_orig = diabetes.copy()
# long insulin is the last actual column containing data in the 
# file, so all following columns should be dropped. this would be columns 21 
# throughn 29. Redefine the DF to remove these columns
diabetes = diabetes.iloc[:,0:21]

# change the date column to an actual date from a string value
diabetes['date'] = pd.to_datetime(diabetes['Date'], format='%m/%d/%Y', errors='coerce')
diabetes.drop(columns='Date',inplace=True)
diabetes.date

# add a day name for later use in models which may require a true or false value
# for a specific day, allowing easy of creatinon for dummy variables
diabetes['day_name'] = diabetes['date'].dt.day_name()

# since there are columns which have NAN values in them, we want to examine how 
# those NAN values are distributed to decide whether they should be dropped from the
# dataset entirely or if we can feel confident in replacing them with some 
# imputed value. If we decide to use an imputed value then we need to 
# make an informed decision on which imputed value to use
diabetes_sdevs = diabetes[['2_day_std','7_day_std','14_day_std','30_day_std','90_day_std']].describe()
diabetes_sdevs_mean_median = diabetes[['2_day_std','7_day_std','14_day_std','30_day_std','90_day_std']].describe().loc[['mean','50%']]
std_modes = diabetes[['2_day_std','7_day_std','14_day_std','30_day_std','90_day_std']].mode()
std_modes.index = ['mode']
sdev_fil_na_value_options = pd.concat([std_modes,diabetes_sdevs_mean_median])
round(sdev_fil_na_value_options,2)

# create a list of columns based on attribute of blood sugar analysis
# to allow for faster iteration over those columns to obtain the stats which 
# will be evaluated for replacement of NaN values in the specified columns
# so that further analysis can be completed without losing to much of the data
sdev_column_list = []
avgs_list = []
times_in_range_list = []

for column in diabetes.columns:
    if column.endswith('std'):
        sdev_column_list.append(column)
    elif column.endswith('avg'):
        avgs_list.append(column)
    elif column.endswith('time_in_range'):
        times_in_range_list.append(column)
        
print("Standard Deviation Columns:", sdev_column_list)
print("Average Columns:", avgs_list)
print("Time in Range Columns:", times_in_range_list)

pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
list_combination = [sdev_column_list, avgs_list, times_in_range_list]
for i in list_combination:
    description = diabetes[i].describe()
    description_mean_median = description.loc[['mean','50%']]
    
    attr_modes = diabetes[i].dropna().mode()
    
    # Handling the case where mode returns more than one mode per column.
    averaged_modes = {}
    for column in i:
        column_modes = attr_modes[column]
        if len(column_modes) >1:  # Check if the column has exactly two modes
            averaged_modes[column] = column_modes.mean()
        else:
            averaged_modes[column] = column_modes[0] if not column_modes.empty else None  # Take the first mode or None if no mode
    
    # Create a DataFrame from the averaged modes dictionary.
    modes_df = pd.DataFrame(averaged_modes, index=['mode'])
    
    # Concatenate the mean, median, and modes DataFrames
    full_description = pd.concat([description_mean_median, modes_df])
    
    # Optional: round the values to 2 decimal places
    full_description_rounded = full_description.round(2)
    
    print(full_description_rounded)

pd.reset_option('all')

# replacing the NaN values with the median values of each column
for i in list_combination:
    diabetes[i] = diabetes[i].fillna(diabetes[i].describe().loc['50%'])
    
# replacing the NaN values in the long insulin category with the median value
diabetes.long_insulin = diabetes.long_insulin.fillna(diabetes.long_insulin.describe().loc['50%'])

# quickly check how many records have 10 units of long ascting insulin
diabetes[diabetes['long_insulin'] != 10]
diabetes[diabetes['long_insulin'] == 10]

# create a correlation dataframe
diabetes_correlation = diabetes.select_dtypes(include='float64').corr()

# plot the correlation dataframe to see how strongly, either positively or negatively
# correlated the various attributes are
# examining the results we find, somewhat unsurprisingly that the previous 
# time in range reading (30-day being previous to 90-day) is highly correlated
# with the following period's time in range reading. As you move over time, the 
# amount of variability in your long term time in range is mostly explained
# by your shorter term time in range. This relationship strength increase over
# time. 2 day explains most of the variability in 7 day and 7 day explains
# even more of the variability in the 14 day, etc.
# the negative correlations show up mosting between the standard deviations and 
# the time in range measures. This should come as no surprise given that as your
# standard deviations increase in any form, the range of values which are possible
# by definition is also increasing. A larger standard deviation thus would 
# more likely than not, relate to a person spending less time in their ideal blood
# sugar range. What is interesting to find, is that there is very little correlation
# between the daily macronutrients consumed and the term (2-, 7-, 14-day, etc.) 
# attribute measures (average, standard deviation, and time in range).
plt.figure(figsize=(30,30)) #set plot figure size
ax = sns.heatmap(diabetes_correlation,annot=True, cmap='crest')
plt.xticks(fontsize=20) # increase the size of the xticks on heatmap
plt.yticks(fontsize=20) # increase the size of the yticks on heatmap


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=24) # increase the size of the text on the colorscale

plt.show()

pd.set_option('display.width',None)
pd.set_option('display.max_columns',None)

macros = ['fat', 'carbs', 'protein'] # create a list of macros to iterate through
time_ranges = [2, 7, 14, 30, 90] # create a list of day ranges which are measured in the dataframe

correlation_data = [] #create an empty list to add correlation tables to

# iterate through the lists of day ranges and macros to produce moving average
# calculations of the macro nutrients and their corresponding day ranges
# an example here is a 2 day moving average of carbs to measure the correlation
# with the 2 day time in range measurement
for macro, days in product(macros, time_ranges):
    # Calculate the moving average for the current macro and time range
    mavg_col_name = f'{days}_day_mavg_{macro}'
    diabetes[mavg_col_name] = diabetes[macro].rolling(window=days).mean()
    
    # Calculate the correlation matrix for the current combination
    corr_matrix = diabetes[[f'{days}_day_time_in_range', mavg_col_name]].dropna().corr()
    
    # Extract the correlation value of interest and add it to the list
    corr_value = corr_matrix.loc[f'{days}_day_time_in_range', mavg_col_name]
    correlation_data.append(((days, macro), corr_value))

# Convert the list of correlations into a DataFrame with MultiIndex
index = pd.MultiIndex.from_tuples([x[0] for x in correlation_data], names=['Days', 'Macro'])
combined_corr_df = pd.DataFrame([x[1] for x in correlation_data], index=index, columns=['Days Time In Range Correlation'])


height_per_row = 0.5  # This means each row will have a height of 1 inch
n_rows = combined_corr_df.shape[0]
fig_height = height_per_row * n_rows  # Total height of the figure

# Create the heatmap with increased figure height
plt.figure(figsize=(10, fig_height))  # Adjust the 10 (width) as needed
ax = sns.heatmap(combined_corr_df, annot=True, cmap='crest')
ax.set_ylabel('Days Moving Average - Macro')
plt.show()


print(combined_corr_df)

ax = sns.heatmap(combined_corr_df,annot=True, cmap='crest')
ax.set_ylabel('Days Moving Average - Macro')
plt.show()

# the correlaiton matrix suggests that increased consumption of carbs and fats 
# will actually (aside from the short term of 2-days) help to improve the time
# spent within the set blood sugar range goal and that this relationship
# increase over the long term, while protein in the short term actually decreases
# the time spent in the goal range, and while this tends to improve and change to a positive
# correlated relationship in the long term, the correlation is still very weak,
# relative to that of carbs and fats. Given that one could not simply eat their way
# to an ideal time in range these correlations can only be taken with something
# of a gain of sale and perhaps, in addition, the correlation of total calories
# should be examined for correlation to times in range, determining if there is
# a stronger correlation to total consumption and time in range than there is in
# any specific macro nutrient. Undersanding ahead of time that there inherently
# will be a correlation becuase with less food there generally would be less insulin
# requirements it might still be interesting to see how much correlaiton there is
# since on the flip side, one cannot simply starve themselves into an ideal time in range

# creating a correlation matrix for the moving average of total calories and
# the time spent in the desired blood sugar range indicates that over time, 
# eating more calories increase the amount of time spent in range
# this on its own holds little merit and should not be taken literally as it would
# possible insinuate that someone could consume themselve into an ideal time in range
# it shold be highlighted that this correlation is very weak, and negative in the short term
# and even over the long term, at 0.339 is still weak and should not lead us to 
# lend much cedence to the practice of consumption to ideal blood sugars
# needless to say, there is far more to explore and undersand when it comes to 
# delivering a conclusion about driving ones overall time spent in the ideal
# blood sugar range and this limited data set is unlikely to provide the insights
# needed
diabetes['total_calories'] =\
    (diabetes['fat']*9) + (diabetes['protein']*4) + (diabetes['carbs']*4)


for day in time_ranges:
    diabetes[f'{day}_day_mavg_total_calories'] = diabetes['total_calories'].rolling(window=day).mean()

for day in time_ranges:
    time_in_range_col = f'{day}_day_time_in_range'
    mavg_col_name = f'{day}_day_mavg_total_calories'

    # Check if the time in range column exists in the DataFrame to avoid KeyError
    if time_in_range_col in diabetes.columns and mavg_col_name in diabetes.columns:
        # Calculate the correlation matrix for the current day's time in range and moving average
        corr_matrix = diabetes[[time_in_range_col, mavg_col_name]].dropna().corr()
        # Append the result to the correlation_data list
        correlation_data.append((day, corr_matrix.iloc[0, 1]))  # .iloc[0, 1] gets the off-diagonal element which is the correlation coefficient

# After collecting all correlations, you can create a DataFrame from them
# Each tuple in the correlation_data list contains a 'day' and the corresponding correlation value
correlation_df = pd.DataFrame(correlation_data, columns=['Days', 'Correlation'])

print(correlation_df)




