import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
data = pd.read_csv('2012-2013 Solar home electricity data v2.csv') 
time_intervals_df = data.iloc[:, 5:-1]
time_intervals_df.columns = pd.to_datetime(time_intervals_df.columns)
time_intervals_df.columns = time_intervals_df.columns.strftime('%H:%M')
data_melted = pd.melt(data, id_vars=['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date'],
                      value_vars=time_intervals_df.columns, var_name='time_interval', value_name='total consumption')

data_melted['time_interval'] = data_melted['time_interval'] + ':00'

data_melted['datetime'] = pd.to_datetime(data_melted['date'], format='%d-%m-%Y') + pd.to_timedelta(data_melted['time_interval'])
data_melted.drop(['date', 'time_interval'], axis=1, inplace=True)


cate_data = data_melted.groupby(['Consumption Category', 'datetime'])['total consumption'].mean().reset_index()


for category in cate_data['Consumption Category'].unique():
category_subset = cate_data[cate_data['Consumption Category'] == category]
plt.plot(category_subset['datetime'], category_subset['total consumption'], label=category)

plt.xlabel('datetime')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption Evolution Over Time')
plt.legend()
plt.show()

for category in cate_data['Consumption Category'].unique():
    category_subset = cate_data[cate_data['Consumption Category'] == category]
    decomposition = seasonal_decompose(category_subset['total consumption'], period=365)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(4, 1, 1)
    plt.plot(category_subset['datetime'], category_subset['total consumption'], label='Original')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(category_subset['datetime'], trend, label='Trend')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(category_subset['datetime'], seasonal, label='Seasonal')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(category_subset['datetime'], residual, label='Residual')
    plt.legend()
    plt.show()


first_data = cate_data[cate_data['datetime'] < '2023-01-01']
test_check_data = cate_data[cate_data['datetime'] >= '2023-01-01']

for category in cate_data['Consumption Category'].unique():
    category_train_data = first_data[first_data['Consumption Category'] == category]
    category_test_data = test_check_data[test_check_data['Consumption Category'] == category]

    model = SARIMAX(category_train_data['total consumption'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 365))
    result = model.fit()
    predictions = result.predict(start=0, end=len(category_test_data)-1)


    plt.plot(category_test_data['datetime'], category_test_data['total consumption'], label='Actual')
    plt.plot(category_test_data['datetime'], predictions, label='Predicted')
    plt.xlabel('datetime')
    plt.ylabel('Energy Consumption')
    plt.title(f'Energy Consumption Prediction for Category {category}')
    plt.legend()
    plt.show()

    rmse = np.sqrt(mean_squared_error(category_test_data['total consumption'], predictions))
    mse = mean_squared_error(category_test_data['total consumption'], predictions)
    mae = mean_absolute_error(category_test_data['total consumption'], predictions)
    mape = np.mean(np.abs((category_test_data['total consumption'] - predictions) / category_test_data['total consumption'])) * 100

    print(f"Category: {category}")
    print(f"RMSE: {rmse}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
