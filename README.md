# EX.NO.09        A project on Time Series Analysis for Website Visitor Forecasting Using the ARIMA Model
### Date: 

### AIM:
To create a project on time series analysis for forecasting website visitor traffic using the ARIMA model in Python and compare its performance with other forecasting models.
### ALGORITHM:
1. Explore the dataset of website visitors
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('/content/daily_website_visitors.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Page.Loads'] = df['Page.Loads'].str.replace(',', '').astype(int)
df.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df['Page.Loads'])
plt.title('Time Series Plot of Page Loads')
plt.xlabel('Date')
plt.ylabel('Page Loads')
plt.show()

def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

adf_test(df['Page.Loads'])

plot_acf(df['Page.Loads'], lags=30)
plot_pacf(df['Page.Loads'], lags=30)
plt.show()

df['Page_Loads_diff'] = df['Page.Loads'].diff().dropna()
plt.figure(figsize=(12, 6))
plt.plot(df['Page_Loads_diff'])
plt.title('Differenced Time Series of Page Loads')
plt.xlabel('Date')
plt.ylabel('Differenced Page Loads')
plt.show()

adf_test(df['Page_Loads_diff'].dropna())

p, d, q = 1, 1, 1
model = ARIMA(df['Page.Loads'], order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=10)
print("Forecasted Page Loads:", forecast)

plt.figure(figsize=(12, 6))
plt.plot(df['Page.Loads'], label='Original Data')
plt.plot(forecast.index, forecast, color='red', label='Forecast', marker='o')
plt.title('ARIMA Model Forecast for Page Loads')
plt.xlabel('Date')
plt.ylabel('Page Loads')
plt.legend()
plt.show()

train_size = int(len(df) * 0.8)
train, test = df['Page.Loads'][:train_size], df['Page.Loads'][train_size:]
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()
predictions = model_fit.forecast(steps=len(test))

mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual Data')
plt.plot(test.index, predictions, color='red', label='Predicted Data')
plt.title('Actual vs Predicted Page Loads')
plt.xlabel('Date')
plt.ylabel('Page Loads')
plt.legend()
plt.show()
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/04efd386-f565-40e1-bf69-06ec45b2f5a9)

![image](https://github.com/user-attachments/assets/b9b5e1f4-e9ac-43cc-8f9a-c55336a23b31)

![image](https://github.com/user-attachments/assets/d1020850-03e5-4fc0-9bce-965b654b5aa4)

![image](https://github.com/user-attachments/assets/498a6f22-43fc-4ff6-9f96-a42a3ca11a8c)

![image](https://github.com/user-attachments/assets/3dd26053-9026-476c-b0b0-6682856860df)

![image](https://github.com/user-attachments/assets/dfd5d18c-fd12-4e9b-b28e-fd68730a221f)

![image](https://github.com/user-attachments/assets/3df44fba-b014-4ce3-817c-1eda02574a5d)

![image](https://github.com/user-attachments/assets/e7d76561-9468-4eb0-a38f-7bd629c7d7d1)

![image](https://github.com/user-attachments/assets/c17872cf-80bf-48d3-a6ed-fc79ac8b719a)

### RESULT:
Thus, the program run successfully based on the ARIMA model using python.
