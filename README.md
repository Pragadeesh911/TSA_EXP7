# Ex.No: 07                                       AUTO REGRESSIVE MODEL


### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = "timeseries_2year.csv"
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
data.set_index('date', inplace=True)
a
print(data.head())
weekly_sales = data['sales'].dropna()

result = adfuller(weekly_sales.dropna())
print(f'\nADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
if result[1] < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")

train_size = int(len(weekly_sales) * 0.8)
train, test = weekly_sales[:train_size], weekly_sales[train_size:]
print(f"\nTrain size: {len(train)} weeks | Test size: {len(test)} weeks")

# Plot ACF and PACF
fig, ax = plt.subplots(2, figsize=(10, 6))
plot_acf(train.dropna(), ax=ax[0], title='Autocorrelation Function (ACF) - Weekly Sales')
plot_pacf(train.dropna(), ax=ax[1], title='Partial Autocorrelation Function (PACF) - Weekly Sales')
plt.tight_layout()
plt.savefig('acf_pacf_plot.png', dpi=150)
plt.show()

ar_model = AutoReg(train.dropna(), lags=8).fit()

# Make predictions on test set
ar_pred = ar_model.predict(
    start=len(train),
    end=len(train) + len(test) - 1,
    dynamic=False
)
ar_pred.index = test.index

plt.figure(figsize=(10, 4))
plt.plot(test, label='Test Data', marker='o', markersize=4)
plt.plot(ar_pred, label='AR Model Prediction', color='red', linestyle='--', marker='x', markersize=4)
plt.title('AR Model Prediction vs Test Data (Weekly Sales)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show()

mse = mean_squared_error(test, ar_pred)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(test.values - ar_pred.values))
print(f'\nModel Performance Metrics:')
print(f'  Mean Squared Error  (MSE) : {mse:.2f}')
print(f'  Root Mean Sq. Error (RMSE): {rmse:.2f}')
print(f'  Mean Absolute Error (MAE) : {mae:.2f}')

plt.figure(figsize=(12, 5))
plt.plot(train, label='Train Data', color='steelblue')
plt.plot(test, label='Test Data', color='orange')
plt.plot(ar_pred, label='AR Model Prediction', color='red', linestyle='--')
plt.axvline(x=train.index[-1], color='gray', linestyle=':', label='Train/Test Split')
plt.title('Train, Test, and AR Model Prediction (Weekly Sales)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.savefig('full_ar_model_plot.png', dpi=150)
plt.show()
```
### OUTPUT:
GIVEN DATA:
<img width="987" height="409" alt="image" src="https://github.com/user-attachments/assets/c796c871-2ff5-4f7e-b377-0a5902fe207e" />

PACF - ACF:
<img width="1224" height="359" alt="image" src="https://github.com/user-attachments/assets/ff451d72-cc52-4f8b-bcbd-b967a70f3077" />
<img width="1214" height="349" alt="image" src="https://github.com/user-attachments/assets/3cf8453d-c615-40c1-afef-09f6f542d0b8" />


PREDICTION:
<img width="1248" height="628" alt="image" src="https://github.com/user-attachments/assets/9d425400-cfee-41fc-a141-fd21cb984035" />

FINIAL PREDICTION:
<img width="1218" height="521" alt="image" src="https://github.com/user-attachments/assets/8b1982ad-0567-4217-b9d0-3da0f28e790f" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
