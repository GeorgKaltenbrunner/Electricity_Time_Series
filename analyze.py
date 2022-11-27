# Imports

from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
import datetime

plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

# Data

df = pd.read_csv('Electric_production.csv', parse_dates=['DATE'], index_col='DATE')
df.rename(columns={'IPG2211A2N': 'production'}, inplace=True)

# ---------------------------------------------------------------------------------------------------------------------
# First insights
df.head()
df.shape
df.describe()
df.info()
print(f"Start of the time series: {min(df.index)}")
print(f"End of the time series: {max(df.index)}")


# ---------------------------------------------------------------------------------------------------------------------
# Visualize
# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='production', dpi=100):
    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


plot_df(df, x=df.index, y=df.production, title='Electricity production from 1985 to 2018')

# Single plot for each year
# Plot Dataset for each year

df.reset_index(inplace=True)

# Prepare data
df['year'] = [d.year for d in df.DATE]
df['month'] = [d.strftime('%b') for d in df.DATE]
years = df['year'].unique()

# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(30, 20), dpi=80)
for i, y in enumerate(years):
    if i > 0:
        plt.plot(df.loc[df['year'] == y]['month'].unique(),
                 df.loc[df['year'] == y]['production'].groupby(by=df['month']).mean(), color=mycolors[i], label=y)
        # plt.plot('month', 'sales', data=df.loc[df.year==y, :], color=mycolors[i], label=y)
        plt.text(df.loc[df.year == y, :].shape[0] - .9, df.loc[df.year == y, 'production'][-1:].values[0], y,
                 fontsize=12, color=mycolors[i])

# Decoration
# plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$production$', xlabel='$Month$') # $ = kursiv
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot Production of Electricity", fontsize=20)
plt.xlabel('$Month$')
plt.ylabel('$production$')
plt.show()

# Boxplots and seasonality
# One boxplot for trend over the years
# One boxplot for sales per month of all years -> see seasonality

# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=80)
sns.boxplot(x='year', y='production', data=df, ax=axes[0])
sns.boxplot(x='month', y='production', data=df.loc[~df.year.isin([1991, 2008]), :])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18);
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
fig.autofmt_xdate()
plt.show()

# Trend and Seasonality only
# Trend, Seasonality

# df = df.set_index('DATE')
result = seasonal_decompose(df['production'], model='multiplicable', period=12)

result.trend.plot(title='Trend')
result.seasonal.plot(title='Seasonality')

# ---------------------------------------------------------------------------------------------------------------------
# Stationality
# ADF Test
result = adfuller(df.production.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# KPSS Test
result = kpss(df.production.values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# plot stationality
# Draw Plot
plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 120})
autocorrelation_plot(df.production.tolist())

# ---------------------------------------------------------------------------------------------------------------------
# Stationality
# Missing values

start = min(df.index)
end = max(df.index)
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

# Check missing values
for index in df.index:
    if index not in date_generated:
        print(index)

# ---------------------------------------------------------------------------------------------------------------------
# Autocorrelation and partial autocorrelation
# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 3), dpi=100)
plot_acf(df.production.tolist(), lags=50, ax=axes[0])
plot_pacf(df.production.tolist(), lags=50, ax=axes[1])

# Lag Plots


plt.rcParams.update({'ytick.left': False, 'axes.titlepad': 10})

fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(df.production, lag=i + 1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i + 1))

fig.suptitle('Lag Plots of Drug Sales', y=1.05)
plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Estimate the forecastability
# Apporximate Entropy

def ApEn(U, m, r):
    """Compute Aproximate entropy"""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m + 1) - _phi(m))


print(ApEn(df.detrended, m=2, r=0.2 * np.std(df.detrended)))


# Sample Extropy

def SampEn(U, m, r):
    """Compute Sample entropy"""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m + 1) / _phi(m))


print(SampEn(df.detrended, m=2, r=0.2 * np.std(df.detrended)))
