import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Read the Excel files for 2022 and 2023 data
df_2022 = pd.read_excel('Native_Load_2022.xlsx')
df_2023 = pd.read_excel('Native_Load_2023.xlsx')

# Convert 'Hour Ending' column to datetime with error handling for 2022 data
df_2022['Hour Ending'] = pd.to_datetime(df_2022['Hour Ending'], format='%m/%d/%Y %H:%M', errors='coerce')

# Drop rows with NaT values and count affected rows for 2022 data
affected_rows_count_2022 = df_2022['Hour Ending'].isna().sum()
df_2022.dropna(subset=['Hour Ending'], inplace=True)

# Extract additional features for 2022 data
df_2022['Hour'] = df_2022['Hour Ending'].dt.hour
df_2022['DayOfWeek'] = df_2022['Hour Ending'].dt.dayofweek
df_2022['Month'] = df_2022['Hour Ending'].dt.month

# Set 'Hour Ending' as the index for 2022 data
df_2022.set_index('Hour Ending', inplace=True)

# Convert 'Hour Ending' column to datetime with error handling for 2023 data
df_2023['Hour Ending'] = pd.to_datetime(df_2023['Hour Ending'], format='%m/%d/%Y %H:%M', errors='coerce')

# Drop rows with NaT values and count affected rows for 2023 data
affected_rows_count_2023 = df_2023['Hour Ending'].isna().sum()
df_2023.dropna(subset=['Hour Ending'], inplace=True)

# Extract additional features for 2023 data
df_2023['Hour'] = df_2023['Hour Ending'].dt.hour
df_2023['DayOfWeek'] = df_2023['Hour Ending'].dt.dayofweek
df_2023['Month'] = df_2023['Hour Ending'].dt.month

# Set 'Hour Ending' as the index for 2023 data
df_2023.set_index('Hour Ending', inplace=True)

# Select the target column for prediction
target_column = 'ERCOT'

# Remove commas and convert numeric columns to float for 2022 data
df_2022[target_column] = df_2022[target_column].replace({',': ''}, regex=True).astype(float)

# Apply seasonal decomposition for 2022 data
seasonal_period = 3  # Assuming a daily (24-hour) seasonality
stl = STL(df_2022[target_column], seasonal=seasonal_period, period=350)
res = stl.fit()
seasonal_component_2022 = res.seasonal

# Split the data into train and test sets for 2022 data
train_size = int(0.8 * len(df_2022))
train_data = df_2022[:train_size]
test_data = df_2022[train_size:]

# Create training set with features for 2022 data
X_train = train_data[['Hour', 'DayOfWeek', 'Month']]
X_train['Seasonal'] = seasonal_component_2022[:train_size].values
y_train = train_data[target_column]

# Create test set with features for 2022 data
X_test = test_data[['Hour', 'DayOfWeek', 'Month']]
X_test['Seasonal'] = seasonal_component_2022[train_size:].values
y_test = test_data[target_column]

# Fit an XGBoost model for 2022 data
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Make predictions on test data for 2022 data
# y_pred_test_2022 = model.predict(X_test)
y_pred_test_2022 = model.predict(X_test)

# Calculate the root mean squared error (RMSE) for test data in 2022
rmse_test_2022 = np.sqrt(mean_squared_error(y_test, y_pred_test_2022))
print('Test RMSE (2022):', rmse_test_2022)

# Print the number of affected rows for 2022 data
print(f"Removed {affected_rows_count_2022} rows that couldn't be parsed as datetime for 2022 data.")

# Apply seasonal decomposition for 2023 data
stl_2023 = STL(df_2023[target_column], seasonal=seasonal_period, period=350)
res_2023 = stl_2023.fit()
seasonal_component_2023 = res_2023.seasonal

# Create feature set for 2023 data
X_2023 = df_2023[['Hour', 'DayOfWeek', 'Month']]
X_2023['Seasonal'] = seasonal_component_2023

# Make predictions on 2023 data
y_pred_2023 = model.predict(X_2023)

# Print the number of affected rows for 2023 data
print(f"Removed {affected_rows_count_2023} rows that couldn't be parsed as datetime for 2023 data.")

# Plot the predicted values for 2022 data
plt.plot(test_data.index, y_test, label='Actual (2022)')
plt.plot(test_data.index, y_pred_test_2022, label='Predicted (2022)', color='r')

# Plot the predicted values for 2023 data
plt.plot(df_2023.index, df_2023[target_column], label='Actual (2023)')
plt.plot(df_2023.index, y_pred_2023, label='Predicted (2023)', color='g')

plt.title('Forecast')
plt.xlabel('Time')
plt.ylabel(target_column)
plt.legend()
plt.show()
