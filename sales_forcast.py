#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


store_sales = pd.read_csv('train.csv')
store_sales.head()


# In[ ]:





# ## check for null values

# In[3]:


store_sales.info()


# ## Dropping store and item columns

# In[4]:


store_sales = store_sales.drop(['store', 'item'], axis = 1)


# ## converting date from object datatype to dateTime datatype

# In[5]:


store_sales['date'] = pd.to_datetime(store_sales['date'])


# In[6]:


store_sales.info()


#  ## Converting date to month period and sum the number of the items in each month

# In[ ]:





# In[7]:


store_sales['date'] = store_sales['date'].dt.to_period('M')
monthly_sales = store_sales.groupby('date').sum().reset_index()


# In[8]:


monthly_sales.head()
# for date, rows in monthly_sales:
#     print(date)
#     print(rows)


# ## Convert the resulting date column to timestamp

# In[9]:


monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales.head(10)


# ## Visualization

# In[10]:


plt.figure(figsize=(15, 5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Monthly Customer Sales')
plt.show()


# ## Call the difference on the sales columns to make the sales data stationery

# In[11]:


monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
# print(monthly_sales)
monthly_sales = monthly_sales.dropna()
monthly_sales.head(10)


# In[31]:


plt.figure(figsize=(15, 5))
plt.plot(monthly_sales['date'], monthly_sales['sales_diff'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Monthly Customer Sales Difference')
plt.show()


# ## Dropping of sales and date

# In[13]:


supervised_data = monthly_sales.drop(['sales', 'date'], axis = 1)


# ## Preparing the supervised data

# In[14]:


for i in range(1, 13):
    col_name = 'month_' + str(i)
    supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop=True)
supervised_data.head()


#  ## Split the data into train and test

# In[15]:


train_data = supervised_data[:-12]
test_data = supervised_data[-12:]
print('Train Data Shape: ', train_data.shape)
print('Test Data Shape: ', test_data.shape)


# In[16]:


scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)


# In[17]:


X_train, y_train = train_data[:, 1:], train_data[:, 0:1]
X_test, y_test = test_data[:, 1:], test_data[:, 0:1]


# In[18]:


y_train = y_train.ravel()
y_test = y_test.ravel()
print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_Test Shape: ', X_test.shape)
print('y_test Shape: ', y_test.shape)


# ## Make prediction data frame to merge the predicted sales prices of all trained algorithms

# In[19]:


sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)


# In[20]:


act_sales = monthly_sales['sales'][-13:].to_list()


# ## To create the linear regression model and predicted output

# In[21]:


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pre = lr_model.predict(X_test)


# In[22]:


lr_pre = lr_pre.reshape(-1, 1)
# this is a set matrix contains the input features of the test data and also the predicted output
lr_pre_test_set = np.concatenate([lr_pre, X_test], axis = 1)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)


# In[23]:


result_list = []
for index in range(0, len(lr_pre_test_set)):
    result_list.append(lr_pre_test_set[index][0] + act_sales[index])
lr_pre_series = pd.Series(result_list, name='Linear Prediction')
predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True)
predict_df.head()


# In[24]:


lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'], monthly_sales['sales'][-12:]))
lr_mae = mean_absolute_error(predict_df['Linear Prediction'], monthly_sales['sales'][-12:])
lr_r2 = r2_score(predict_df['Linear Prediction'], monthly_sales['sales'][-12:])
print('Linear Regression MSE: ', lr_mse)
print('Linear Regression MAE: ', lr_mae)
print('Linear Regression R2_Score: ', lr_r2)


# ## Visualization of the prediction against each actual sales

# In[30]:


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


# In[ ]:


# Create the beam analysis object
from anastruct import SystemElements # the class that will be used to create the finite element analysis object

beam_analysis = SystemElements()
EI = 5000

# Define the elements that make up the beam and set their cross section

beam_analysis.add_element(location=[[0,0],[6, 0]], EI=EI)
beam_analysis.add_element(location=[[6,0],[11, 0]], EI=EI)
beam_analysis.add_element(location=[[11,0],[17, 0]], EI=EI)

# Add support conditions
beam_analysis.add_support_hinged(node_id=1)
beam_analysis.add_support_roll(node_id=2)
beam_analysis.add_support_roll(node_id=3)
beam_analysis.add_support_hinged(node_id=4)

# Add loadings on the beam
beam_analysis.q_load(element_id=1, q=-40)
beam_analysis.q_load(element_id=2, q=-20)
beam_analysis.q_load(element_id=3, q=-40)

# Display the beam structure using method show_structure
beam_analysis.show_structure(figsize=(18, 18))

# Compute the response of the beam due to loadings using solve method
beam_analysis.solve()

# Plot the bending moment using show_bending_moment method
beam_analysis.show_bending_moment(figsize=(18, 18))

# Plot the shearing force diagram using show_shear_force method
beam_analysis.show_shear_force(figsize=(18, 18))


# In[ ]:


pip install anastruct


# In[ ]:


get_ipython().system('pip install anastruct')


# In[ ]:


import numpy as np


# In[ ]:


arr1 = np.array([23, 10, 20, 30, 24, 15]).reshape(3,2)
arr2 = np.where(arr1 % 2 == 0, arr1, 0)
arr2


# In[ ]:


print(np.var(arr1))


# In[ ]:


print(id(arr1))


# In[ ]:


arr2_ref = arr1


# In[ ]:


arr2_ref[2] = 600


# In[ ]:


arr2_ref


# In[ ]:


arr1


# In[ ]:


print(id(arr2_ref))


# In[ ]:


years = list(range(1991, 2022))


# In[ ]:


# Dowload from a website
url_start = "https://www.basketball-reference.com/awards/awards_{}.html"


# In[ ]:


import requests
# year = 1991
# url = url_start.format(year)
# data = requests.get(url)
# data.text
for year in years:
    url = url_start.format(year)
    data = requests.get(url)
    # i added encoding='utf-8'to open function
    with open('mvps/{}.html'.format(year), 'w+', encoding='utf-8') as f:
        f.write(data.text)


# In[ ]:


# !pip install beautifulsoup4


# In[ ]:


from bs4 import BeautifulSoup


# In[ ]:


with open('mvps/1991.html', encoding='utf-8') as f:
    page = f.read()
soup = BeautifulSoup(page, 'html.parser')
soup.find('tr', class_='over_header').decompose()


# In[ ]:


mvp_table = soup.find(id='mvp')


# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_html(str(mvp_table))[0] # dataframe of the first element
df


# In[ ]:


df = []
for year in years:
    
    with open(f'mvps/{year}.html', encoding='utf-8') as f:
        page = f.read()
    soup = BeautifulSoup(page, 'html.parser')
    soup.find('tr', class_='over_header').decompose()
    mvp_table = soup.find(id='mvp')
    mvp = pd.read_html(str(mvp_table))[0]# dataframe of the first element
    mvp['year'] = year
    df.append(mvp)


# In[ ]:


mvps = pd.concat(df)
mvps.tail()


# In[ ]:


mvps.to_csv('mvps.csv')


# In[ ]:


import requests
player_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"
url = player_stats_url.format(1991)
data = requests.get(url)
with open('players/1991.html', 'w+', encoding='utf-8') as f:
    f.write(data.text)


# In[ ]:


player_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"


# In[ ]:


# !pip install selenium


# In[ ]:


from selenium import webdriver
from selenium.webdriver.edge.service import Service # i solve this


# In[ ]:


# web_driver = webdriver.Edge('msedgedriver.exe')
service = Service('msedgedriver.exe')

web_driver = webdriver.Edge(service=service)


# In[ ]:


web_driver


# In[ ]:


import time
year = 1991
url = player_stats_url.format(year)
web_driver.get(url)
# web_driver.execute_script('window.scrollTo(1, 100000)')
web_driver.execute_script('window.scrollTo(1, 10000)')
time.sleep(2)
html = web_driver.page_source


# In[ ]:


with open('players/{}.html'.format(year), 'w+', encoding='utf-8') as f:
    f.write(html)


# In[ ]:





# In[ ]:


years = list(range(1991, 2022))


# In[ ]:


import time
for year in years:
    url = player_stats_url.format(year)
    web_driver.get(url)
    # web_driver.execute_script('window.scrollTo(1, 100000)')
    web_driver.execute_script('window.scrollTo(1, 10000)')
#     time.sleep(2)
    
    html = web_driver.page_source
    with open('players/{}.html'.format(year), 'w+', encoding='utf-8') as f:
        f.write(html)

# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# # ...

# for year in years:
#     url = player_stats_url.format(year)
#     web_driver.get(url)
    
#     # Wait for an element to be present on the page (adjust timeout as needed)
#     WebDriverWait(web_driver, 10).until(
#         EC.presence_of_element_located((By.XPATH, "your_xpath_here"))
#     )
    
#     # Alternatively, wait for the page to be fully loaded using document.readyState
#     WebDriverWait(web_driver, 10).until(
#         lambda driver: driver.execute_script("return document.readyState") == "complete"
#     )
    
#     html = web_driver.page_source
#     with open(f'players/{year}.html', 'w+', encoding='utf-8') as f:
#         f.write(html)


# In[ ]:


df = []
for year in years:
    with open('players/{}.html'.format(year), encoding='utf-8') as f:
        page = f.read()
    soup = BeautifulSoup(page, 'html.parser')
    soup.find('tr', class_='thead').decompose()
    player_table = soup.find(id='per_game_stats')
    player = pd.read_html(str(player_table))[0]# dataframe of the first element
    player['year'] = year
    df.append(player)


# In[ ]:


players = pd.concat(df)


# In[ ]:


players.head()


# In[ ]:


players.to_csv('players.csv')


# In[ ]:


team_stats_url = 'https://www.basketball-reference.com/leagues/NBA_{}_standings.html'


# In[ ]:


import requests


# In[ ]:


for year in years:
    url = team_stats_url.format(year)
    data = requests.get(url)
    with open('team/{}.html'.format(year), 'w+', encoding='utf-8') as f:
        f.write(data.text)


# In[ ]:


dfs = []
for year in years:
    with open('team/{}.html'.format(year), encoding='utf-8') as f:
        page = f.read()

    soup = BeautifulSoup(page, 'html.parser')
#     soup.find('tr', class_='thead').decompose()
    team_table = soup.find(id='divs_standings_E')
    team = pd.read_html(str(team_table))[0]# dataframe of the first element
    team['year'] = year
    team['team'] = team['Eastern Conference']
    del team['Eastern Conference']
    dfs.append(team)
    
    soup = BeautifulSoup(page, 'html.parser')
#     soup.find('tr', class_='thead').decompose()
    team_table = soup.find(id='divs_standings_W')
    team = pd.read_html(str(team_table))[0]# dataframe of the first element
    team['year'] = year
    team['team'] = team['Western Conference']
    del team['Western Conference']
    dfs.append(team)


# In[ ]:


df = pd.concat(dfs)


# In[ ]:


df.to_csv('team.csv')


# In[ ]:




