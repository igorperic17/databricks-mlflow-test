import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Read the Excel file
df = pd.read_excel('Native_Load_2022.xlsx')

# Convert 'Hour Ending' column to datetime with error handling
df['Hour Ending'] = pd.to_datetime(df['Hour Ending'], format='%m/%d/%Y %H:%M', errors='coerce')

# Drop rows with NaT values and count affected rows
affected_rows_count = df['Hour Ending'].isna().sum()
df.dropna(subset=['Hour Ending'], inplace=True)

# Extract additional features
df['Hour'] = df['Hour Ending'].dt.hour
df['DayOfWeek'] = df['Hour Ending'].dt.dayofweek
df['Month'] = df['Hour Ending'].dt.month

# Set 'Hour Ending' as the index
df.set_index('Hour Ending', inplace=True)

# Select the target column for prediction
target_column = 'ERCOT'

# Remove commas and convert numeric columns to float
df[target_column] = df[target_column].replace({',': ''}, regex=True).astype(float)

# Split the data into train and test sets
train_size = int(0.8 * len(df))
train_data = df[:train_size]
test_data = df[train_size:]

# Create training set with timestamp and engineered features
X_train = train_data[['Hour', 'DayOfWeek', 'Month']]
y_train = train_data[target_column]

# Create test set with timestamp and engineered features
X_test = test_data[['Hour', 'DayOfWeek', 'Month']]
y_test = test_data[target_column]

# Fit an XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

# Print the number of affected rows
print(f"Removed {affected_rows_count} rows that couldn't be parsed as datetime.")

# Plot the predicted values
plt.plot(test_data.index, y_test, label='Actual')
plt.plot(test_data.index, y_pred, label='Predicted', color='r')
plt.title('Forecast')
plt.xlabel('Time')
plt.ylabel(target_column)
plt.legend()
plt.show()
