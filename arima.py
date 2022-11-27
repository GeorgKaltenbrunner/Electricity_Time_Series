# Imports

import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# ---------------------------------------------------------------------------------------------------------------------
# Data
df = pd.read_csv(r'detrendend.csv')
df_train = df.iloc[:int(len(df) * 0.7)]
df_test = df.iloc[int(len(df) * 0.7):]

# ---------------------------------------------------------------------------------------------------------------------
# Find 'd'!
# Original, 1. Diff, 2. Diff
# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(df_train.production);
ax1.set_title('Original Series');
ax1.axes.xaxis.set_visible(False)
# 1st Differencing
ax2.plot(df_train.production.diff());
ax2.set_title('1st Order Differencing');
ax2.axes.xaxis.set_visible(False)
# 2nd Differencing
ax3.plot(df_train.production.diff().diff());
ax3.set_title('2nd Order Differencing')
plt.show()

# ACF
# Draw Plot
plot_acf(df_train.production.tolist(), lags=50, title='First ACF')
plot_acf(df_train.production.diff().dropna(), title='First ACF - Diff')
plot_acf(df_train.production.diff().diff().dropna(), title='Second ACF - Diff')

# ---------------------------------------------------------------------------------------------------------------------
# Find 'p'!

plot_pacf(df_train.production.diff().dropna())

# ---------------------------------------------------------------------------------------------------------------------
# Find 'q'!
plot_acf(df_train.production.diff().dropna())

# ---------------------------------------------------------------------------------------------------------------------
# ARIMA model

model = sm.tsa.arima.ARIMA(df_train.production, order=(2, 1, 2))
model_fit = model.fit()
model_fit.summary()

# Make prediction

pred = model_fit.predict()
plt.show()

df.DATE = pd.to_datetime(df.DATE)
df = df.set_index('DATE')

# Predcit test data

prediciton = model_fit.predict(start=min(df_test.index), end=max(df_test.index))

plt.plot(df_train.index, df_train.production, label='actual')
plt.plot(df_train.index, pred, color='red', label='predicted')
plt.plot(df_test.index, df_test.production, color='green', label='test')
plt.plot(df_test.index, prediciton, color='orange', label='predicted_test')
plt.xlabel('Year')
plt.ylabel('Production')
plt.title('Actual vs predicted Electrcity production per year')
plt.legend()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Validation
# MAE

df_test['pred'] = prediciton
df_test['diff'] = abs(df_test.production - df_test.pred)
MAE = sum(df_test['diff']) / len(df_test)
print(MAE)

# Mean Squared Error
MSE = mean_squared_error(df_test.production, df_test.pred)
print('MSE: %f' % MSE)

# Root mean square error
RMSE = sqrt(MSE)
print('RMSE: %f' % RMSE)
