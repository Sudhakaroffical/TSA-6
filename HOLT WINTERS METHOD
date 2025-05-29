### Name  : SUDHAKAR K
### Reg.No: 212222240107
### Date  : 

# Ex.No: 6               HOLT WINTERS METHOD
### AIM:
   To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. Load and resample the seattle weather data to monthly frequency, selecting the 'Prcp' column.
2. Scale the data using Minmaxscaler then split into training (80%) and testing (20%) sets.
3. Fit an additive Holt-Winters model to the training data and forecast on the test data.
4. Evaluate model performance using MAE and RMSE, and plot the train, test, and prediction results.
5. Train a final multiplicative Holt-Winters model on the full dataset and forecast seattle weather.
### PROGRAM:
```
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
data = pd.read_csv('/content/seattle_weather_1948-2017.csv', index_col='DATE', parse_dates=True)
data = data['PRCP']
data_monthly = data.resample('MS').mean()
scaler = MinMaxScaler()
data_scaled = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(), index=data_monthly.index)
train_data = data_scaled[:int(len(data_scaled) * 0.8)]
test_data = data_scaled[int(len(data_scaled) * 0.8):]
fitted_model_add = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()
test_predictions_add = fitted_model_add.forecast(len(test_data))

# Evaluate performance
print("MAE (Additive):", mean_absolute_error(test_data, test_predictions_add))
print("RMSE (Additive):", mean_squared_error(test_data, test_predictions_add, squared=False))

plt.figure(figsize=(12, 8))
plt.plot(train_data, label='TRAIN', color='black')
plt.plot(test_data, label='TEST', color='green')
plt.plot(test_predictions_add, label='PREDICTION', color='red')
plt.title('Train, Test, and Additive Holt-Winters Predictions')
plt.legend(loc='best')
plt.show()
# Check for zero or negative values in data_scaled
print("Number of zero values:", (data_scaled == 0).sum())
print("Number of negative values:", (data_scaled < 0).sum())

# If there are zero or negative values, consider one of the following:

# 1. Add a small constant to all values
data_scaled = data_scaled + 1e-6 

# 2. Use a different trend/seasonal component (e.g., 'add' instead of 'mul')
final_model = ExponentialSmoothing(data_scaled, trend='add', seasonal='add', seasonal_periods=12).fit()

# 3. If applicable, investigate and handle the zero/negative values in the original data before scaling

# Forecast future values
forecast_predictions = final_model.forecast(steps=12)

# Plot actual data and forecasted values
plt.figure(figsize=(12, 8))
data_scaled.plot(legend=True, label='Current PRCP')
forecast_predictions.plot(legend=True, label='Forecasted PRCP')
plt.xlabel('Date')
plt.ylabel('Precipitation (Scaled)')
plt.title('Precipitation Forecast using Holt-Winters')
plt.show()
```

### OUTPUT:

![image](https://github.com/user-attachments/assets/51689f03-3e7d-4926-a67c-d99b51374ec9)



#### TEST_PREDICTION

![image](https://github.com/user-attachments/assets/40f83f96-e7c7-496b-bcc2-e7ec930e365b)


#### FINAL_PREDICTION

![image](https://github.com/user-attachments/assets/7425d895-a324-4b26-9a83-e1ad040f921a)


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
