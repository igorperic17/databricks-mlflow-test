import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Read the Excel file
df = pd.read_excel('Native_Load_2022.xlsx')

# Modify 'Hour Ending' column to handle "24:00" value
# df['Hour Ending'] = df['Hour Ending'].str.replace(' 24:00', ' 00:00')

# Convert 'Hour Ending' column to datetime with error handling
df['Hour Ending'] = pd.to_datetime(df['Hour Ending'], format='%m/%d/%Y %H:%M', errors='coerce')

# Drop rows with NaT values and count affected rows
affected_rows_count = df['Hour Ending'].isna().sum()
df.dropna(subset=['Hour Ending'], inplace=True)

# Extract additional features
df['Hour'] = df['Hour Ending'].dt.hour
df['DayOfWeek'] = df['Hour Ending'].dt.dayofweek
df['Month'] = df['Hour Ending'].dt.month

# Reset the index
df.reset_index(drop=True, inplace=True)

# Select the target column for prediction
target_column = 'ERCOT'

# Remove commas and convert numeric columns to float
df[target_column] = df[target_column].replace({',': ''}, regex=True).astype(float)

# Apply seasonal decomposition
stl = STL(df[target_column], seasonal=23*31, period=24)  # Assuming a daily (24-hour) seasonality
res = stl.fit()
trend = res.trend
seasonal = res.seasonal
residual = res.resid

# Plot the decomposed components
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df['Hour Ending'], df[target_column], label='Original')
plt.title('Original Data')
plt.xlabel('Time')
plt.ylabel(target_column)
plt.subplot(412)
plt.plot(df['Hour Ending'], trend, label='Trend')
plt.title('Trend Component')
plt.xlabel('Time')
plt.ylabel('Trend')
plt.subplot(413)
plt.plot(df['Hour Ending'], seasonal, label='Seasonal')
plt.title('Seasonal Component')
plt.xlabel('Time')
plt.ylabel('Seasonal')
plt.subplot(414)
plt.plot(df['Hour Ending'], residual, label='Residual')
plt.title('Residual Component')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.tight_layout()
plt.show()

# Split the data into train and test sets
train_size = int(0.8 * len(df))
train_data = df[:train_size][target_column]
test_data = df[train_size:][target_column]

# Use the average seasonal component for prediction
seasonal_average = seasonal[:train_size].mean()
forecast = trend[train_size:] + seasonal_average

# Calculate the root mean squared error (RMSE)
mse = np.mean((forecast - test_data) ** 2)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Print the number of affected rows
print(f"Removed {affected_rows_count} rows that couldn't be parsed as datetime.")

# Plot the predicted values
plt.plot(df['Hour Ending'][train_size:], test_data, label='Actual')
plt.plot(df['Hour Ending'][train_size:], forecast, label='Predicted', color='r')
plt.title('Forecast')
plt.xlabel('Time')
plt.ylabel(target_column)
plt.legend()
plt.show()
