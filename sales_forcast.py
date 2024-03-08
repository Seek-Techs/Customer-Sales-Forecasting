import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow. keras.callbacks import EarlyStopping, ModelCheckpoint

store_sales = pd.read_csv('train.csv')
store_sales.head()

# ## check for null values
store_sales.info()

# ## Dropping store and item columns
store_sales = store_sales.drop(['store', 'item'], axis = 1)

# ## converting date from object datatype to dateTime datatype
store_sales['date'] = pd.to_datetime(store_sales['date'])

store_sales.info()

#  ## Converting date to month period and sum the number of the items in each month
store_sales['date'] = store_sales['date'].dt.to_period('M')
monthly_sales = store_sales.groupby('date').sum().reset_index()

monthly_sales.head()

# ## Convert the resulting date column to timestamp
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales.head(10)
# ## Visualization
plt.figure(figsize=(15, 5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Monthly Customer Sales')
plt.show()

# ## Call the difference on the sales columns to make the sales data stationery
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
# print(monthly_sales)
monthly_sales = monthly_sales.dropna()
monthly_sales.head(10)

plt.figure(figsize=(15, 5))
plt.plot(monthly_sales['date'], monthly_sales['sales_diff'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Monthly Customer Sales Difference')
plt.show()

# ## Dropping of sales and date
supervised_data = monthly_sales.drop(['sales', 'date'], axis = 1)

# ## Preparing the supervised data
for i in range(1, 13):
    col_name = 'month_' + str(i)
    supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop=True)
supervised_data.head()

train_data = supervised_data[:-12]
test_data = supervised_data[-12:]
print('Train Data Shape: ', train_data.shape)
print('Test Data Shape: ', test_data.shape)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

X_train, y_train = train_data[:, 1:], train_data[:, 0:1]
X_test, y_test = test_data[:, 1:], test_data[:, 0:1]

y_train = y_train.ravel()
y_test = y_test.ravel()
print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_Test Shape: ', X_test.shape)
print('y_test Shape: ', y_test.shape)

# ## Make prediction data frame to merge the predicted sales prices of all trained algorithms
sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)
act_sales = monthly_sales['sales'][-13:].to_list()

# ## To create the linear regression model and predicted output
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pre = lr_model.predict(X_test)
lr_pre = lr_pre.reshape(-1, 1)
# this is a set matrix contains the input features of the test data and also the predicted output
lr_pre_test_set = np.concatenate([lr_pre, X_test], axis = 1)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

result_list = []
for index in range(0, len(lr_pre_test_set)):
    result_list.append(lr_pre_test_set[index][0] + act_sales[index])
lr_pre_series = pd.Series(result_list, name='Linear Prediction')
predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True)
predict_df.head()

lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'], monthly_sales['sales'][-12:]))
lr_mae = mean_absolute_error(predict_df['Linear Prediction'], monthly_sales['sales'][-12:])
lr_r2 = r2_score(predict_df['Linear Prediction'], monthly_sales['sales'][-12:])
print('Linear Regression MSE: ', lr_mse)
print('Linear Regression MAE: ', lr_mae)
print('Linear Regression R2_Score: ', lr_r2)

# ## Visualization of the prediction against each actual sales
plt.figure(figsize=(15, 5))
# Actual sales
plt.plot(monthly_sales['date'], monthly_sales['sales'])
#Predicted Sales
plt.plot(predict_df['date'], predict_df['Linear Prediction'])
plt.title('Customer sales Forcast using LR Model', size=15)
plt.xlabel('Date', size=15)
plt.ylabel('Sales', size=15)
plt.legend(['Actual Sales', 'Predicted sales'])
plt.plot()