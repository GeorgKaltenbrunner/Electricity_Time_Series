# Imports

from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
import statsmodels.api as sm

# ---------------------------------------------------------------------------------------------------------------------
# Data
df = pd.read_csv('detrendend.csv', parse_dates=['DATE'], index_col='DATE')

# Transform Month in into number of number int
month_dict = dict()
month_dict['Jan'] = 1
month_dict['Feb'] = 2
month_dict['Mar'] = 3
month_dict['Apr'] = 4
month_dict['May'] = 5
month_dict['Jun'] = 6
month_dict['Jul'] = 7
month_dict['Aug'] = 8
month_dict['Sep'] = 9
month_dict['Oct'] = 10
month_dict['Nov'] = 11
month_dict['Dec'] = 12

month_number_list = []

for i in df.month:
    month_number_list.append(month_dict.get(i))
df['month_number'] = month_number_list

df_train = df.iloc[:int(len(df) * 0.7)]
df_test = df.iloc[int(len(df) * 0.7):]

# ---------------------------------------------------------------------------------------------------------------------
# Linear Regression Model
X = df_train[['year', 'month_number']]
y = df_train.production
# Fit model
model = sm.OLS(y, X).fit()

df_train['predicted_train'] = model.predict(X)

# Plot
plt.plot(df_train.index, df_train.production)
plt.plot(df_train.index, df_train.predicted_train)

# Test prediction
X_test = df_test[['year', 'month_number']]
y_test = df_test.production
# Fit model
model = sm.OLS(y_test, X_test).fit()

df_test['predicted_test'] = model.predict(X_test)

# Plot test and test_prediction
plt.plot(df_test.index, df_test.production)
plt.plot(df_test.index, df_test.predicted_test)

# ---------------------------------------------------------------------------------------------------------------------
# Validation
# MAE_linear_regeression
df_test['diff'] = abs(df_test['production'] - df_test.predicted_test)
MAE = sum(df_test['diff']) / len(df_test)
print(MAE)

# Mean Squared Error
MSE = mean_squared_error(df_test.production, df_test.predicted_test)
print('MSE: %f' % MSE)

# Root mean square error
RMSE = sqrt(MSE)
print('RMSE: %f' % RMSE)
