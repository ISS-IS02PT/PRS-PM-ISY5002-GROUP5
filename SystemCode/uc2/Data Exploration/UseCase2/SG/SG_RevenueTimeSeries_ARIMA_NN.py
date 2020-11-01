# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Parkway Project Use Case 2 (Time Series Revenue Prediction)

# Imports
import pandas as pd
import matplotlib.pyplot as plt
#from pandas import datetime
#from matplotlib import pyplot

# Load the CSV file
CSV_FILE = 'SG_HospRevenue_2017_2019_ByWeek.csv'
series = pd.read_csv(CSV_FILE, parse_dates=True, index_col=0)
print(series.head())

# Plot the time series data
series.plot(figsize=(15, 8), marker='x', title='Singapore Resident Patient Weekly Revenue Number')
plt.xlabel('DATE')
plt.ylabel('REVENUE')
plt.show()

# Autocorrelation
#pd.plotting.autocorrelation_plot(series)
#plt.figure(figsize=(15, 8))
#plt.show()
from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(figsize=(15, 8))
plot_acf(series['REVENUE'], ax=ax)
ax.set_ylabel('Autocorrelation')
ax.set_xlabel('Lag (weeks)')
plt.show()
# from the picture below we choose the first 6 lags

# ## Start of Arima Model

# +
# ARIMA Model
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(series, order=(2,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# -

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

# +
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# Rolling Forecast ARIMA model
X = series.values
size = int(len(X)*0.66)
train, test = X[0:size], X[size:len(X)]
print('train size is:', size)
print('test size is:', len(X)-size)
history = [x for x in train]
predictions = list()
for t in range (len(test)):
    model = ARIMA(history,order=(6,0,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat,obs))
error_mse = mean_squared_error(test,predictions)
error_mae = mean_absolute_error(test,predictions)
print('Test MSE: %.3f' % error_mse)
print('Test MAE: %.3f' % error_mae)

# Plot the error
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

# -

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(test, label='Truth', marker='o')
ax.plot(predictions, label='Arima Predictions', marker='x')
ax.legend()
plt.show()
print('test data (actual) type is ', type(test))
print('predictions data type is', type(predictions))

# +
import numpy as np
# Function to show various error measurement
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    mse = np.mean((forecast - actual)**2)  # MSE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'mse':mse, 'rmse':rmse})

forecast_accuracy(predictions, test)
# -

# ## End of ARIMA model

# ## Start of Neural Net Model

# + colab={"base_uri": "https://localhost:8080/", "height": 72} colab_type="code" id="SD5hwKwkRG9j" outputId="9480b450-b04d-48e3-9777-d8a9dde43c2b"
# visualisation
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# dataset downloading
import io
import requests
import zipfile

# data processing
import pandas as pd

# models
from sklearn.model_selection import train_test_split
import tensorflow as tf

# for saving the processed dataset
import pickle

plt.style.use('seaborn-whitegrid')

print("Tensorflow version: ", tf.__version__)
print(tf.test.gpu_device_name())

# + [markdown] colab_type="text" id="vBTeSSrs54T6"
# Update `CSV_FILE` to use the correct .csv filename. Some zip files contain multiple datasets.

# + colab={"base_uri": "https://localhost:8080/", "height": 224} colab_type="code" id="rJ9_sDXa3Clm" outputId="e0de6543-6792-4394-87be-0a1532ec5beb"
# Note: update CSV_FILE to the .csv filename from above
CSV_FILE = 'SG_HospRevenue_2017_2019_ByWeek.csv'

df = pd.read_csv(CSV_FILE, parse_dates=True, index_col=0)
df.head()

# + [markdown] colab_type="text" id="WAV-3NXu6D4q"
# ### Data Exploration
#
# 1. Plot the dataset
# 2. Compute the min, max, etc
# 3. Plot the autocorrelation

# + colab={"base_uri": "https://localhost:8080/", "height": 508} colab_type="code" id="aaNbc7O35jC8" outputId="0963c90c-038c-47dc-db2c-b1a512dfa05f"
df['REVENUE'].plot(figsize=(15, 8), marker='x', title='Singapore Resident Patient Weekly Revenue Number')
plt.xlabel('DATE')
plt.ylabel('REVENUE')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 175} colab_type="code" id="G7VZO40GBNPc" outputId="6a663e75-663f-4399-b2f0-a42413321584"
df['REVENUE'].describe()

# + colab={"base_uri": "https://localhost:8080/", "height": 508} colab_type="code" id="0Q-Nl1rD5jVR" outputId="232e1bb0-aafd-40ca-d944-2099282766e2"
fig, ax = plt.subplots(figsize=(15, 8))
plot_acf(df['REVENUE'], ax=ax)
ax.set_ylabel('Autocorrelation')
ax.set_xlabel('Lag (weeks)')
plt.show()

# + colab={} colab_type="code" id="AEhS04ocCZ21"
window_size = 2 # largest number of lags above the 95% confidence band

# + [markdown] colab_type="text" id="tZMZoF3n6bE4"
# ### Windowing
#
# 1. Create shifted windows of the dataset.
# 2. Use this to setup our inputs and target.

# + colab={"base_uri": "https://localhost:8080/", "height": 245} colab_type="code" id="zYRQijG9Es2i" outputId="64d156a2-aa09-461c-82f1-5d0e361f2cb9"
# original dataset
df['REVENUE']

# + colab={"base_uri": "https://localhost:8080/", "height": 245} colab_type="code" id="q5AyPLch5jAd" outputId="cbd51fdb-1ea2-43e6-b324-ba053e9dbe56"
# shift up 1 step in time using -1
# (note the date index does not change, we'll fix that later)
df['REVENUE'].shift(-1)

# + colab={"base_uri": "https://localhost:8080/", "height": 245} colab_type="code" id="udcHea4zEGHu" outputId="499bcab4-8a3d-4b0e-dbe8-c22f2536bbf1"
# shift up in time using -2
# (note the quarter index does not change, we'll fix that later)
df['REVENUE'].shift(-2)
# -

# shift up in time using -3
# (note the quarter index does not change, we'll fix that later)
df['REVENUE'].shift(-3)

# + colab={"base_uri": "https://localhost:8080/", "height": 429} colab_type="code" id="oCbIBhsvE4N7" outputId="2002ed0f-ee7f-4999-e6d0-59bf1701ad6f"
# column-wise concatenation

# List comprehension, equivalent to:
# new_columns = []
# for i in range(window_size):
#    new_columns.append(df['value'].shift(-i))
new_columns = [df['REVENUE'].shift(-i) for i in range(window_size+1)]
new_column_names = [f't+{i}' for i in range(window_size+1)]

df_windowed = pd.concat(new_columns, axis=1)
df_windowed.columns = new_column_names
df_windowed

# + colab={"base_uri": "https://localhost:8080/", "height": 429} colab_type="code" id="Nss-QHwcG7f9" outputId="cb517c13-88e9-4a58-d464-2bf92d3d3abf"
# cleanup the NaN at the bottom of the dataset
df_windowed.dropna(inplace=True)
df_windowed

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="y7NBIASe6m6N" outputId="69994189-9282-411f-dcf7-15403a4dfa1c"
# Formulate our problem

df_windowed.columns

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="whMf2snyHH0C" outputId="53615cfd-fd6c-435e-b0ad-d616004c60b2"
# the target we want to predict (lowercase y is a convention for a vector)
y = df_windowed['t+2']

# the input data (uppercase X is a convention for a matrix)
X = df_windowed.drop(columns=['t+2'])

X.shape, y.shape

# + [markdown] colab_type="text" id="q8Zy8Ez36nlj"
# ### Neural Network
#
# 1. Create a neural network using Tensorflow-Keras
# 2. Split the dataset into training and test sets
# 3. Train the neural network using the training set.
# 4. Evaluate the neural network using the test set.
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 227} colab_type="code" id="Iv6XG7Hb6m48" outputId="c507bcc0-37a0-4769-9982-2ee6040283f2"
# https://www.tensorflow.org/guide/keras/overview
# Create a simple Neural Network with 2 Dense Layers
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(16, input_shape=(2,), activation='relu'))
model.add(layers.Dense(1))
model.summary()

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="AIo_kr226s3t" outputId="6cdf884f-971a-4cac-f795-1a12c9160ee5"
#X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle = False)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# + colab={"base_uri": "https://localhost:8080/", "height": 429} colab_type="code" id="zQv6v9mM5i7g" outputId="6149f417-e2f0-442c-9fb6-d93c034b614f"
# Note that the data is shuffled in time
# This is okay because we already preserved the history!
#
# If this bothers you, you can use train_test_split(X, y, shuffle=False)
# It does affect how the Neural Network is initialised, but since we
# will be go through the dataset multiple rounds, it doesn't really matter.
#
# It may matter if your dataset is huge, because the Neural Network training
# gets more chances see the older data. 
X_train

# + colab={"base_uri": "https://localhost:8080/", "height": 245} colab_type="code" id="sVFyKc2tKn0Q" outputId="975a0009-ea44-4a6e-bda5-92038a2085c9"
# What matters is that each row of X_train and each row of y_train
# are for the same quarter!
y_train

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="lN8PsewfKp3X" outputId="37a872ef-2387-4b8a-81f5-21e034abc842"
# train!
# Note: if you have more data, you can split the dataset 3 ways:
#  train, validation, test
# And then use the validation set (e.g. X_val, y_val) in validation_data

model.compile(optimizer='Adam', loss='mse', metrics=['mape'])
history = model.fit(X_train, y_train, batch_size=8, epochs=250,
                    validation_data=(X_test, y_test))

# + colab={"base_uri": "https://localhost:8080/", "height": 291} colab_type="code" id="slgKH-bqcWwe" outputId="e918d0d2-117f-43cf-f50b-e8b5503bcd98"
# Check for overfitting, which is when val_loss starts to go up but
# loss stays decreases or stays constant.

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Learning curve')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# + colab={} colab_type="code" id="6IvVquwzKqQD"
model.save('model_sg_revenue_prediction.h5')

# + colab={"base_uri": "https://localhost:8080/", "height": 429} colab_type="code" id="2t24z4WzOSG2" outputId="0282cce6-ef89-417d-8de6-19a40c62a74d"
# Get a prediction from our model for our data and plot it against the truth

y_pred = model.predict(X)

df_pred = pd.DataFrame(index=X.index, data={'predictions': y_pred.ravel()})
df_pred
# -

type(y_pred)

# + colab={"base_uri": "https://localhost:8080/", "height": 479} colab_type="code" id="ppUviuuZO-oQ" outputId="2b858b5a-b3e5-4dad-f358-391e48803eb4"
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(y, label='Truth', marker='o')
ax.plot(df_pred, label='Neural Network Predictions', marker='x')
ax.legend()
plt.show()

# + colab={} colab_type="code" id="9e_PkhlujHRe"
full_id_with_y_and_pred = pd.concat([X,y,df_pred],axis=1)
full_id_with_y_and_pred.to_csv('full_sg_revenue_with_y_and_pred.csv')

# +
# Get a prediction from our model for our data and plot it against the truth

y_pred_test = model.predict(X_test)

df_pred_test = pd.DataFrame(index=X_test.index, data={'predictions': y_pred_test.ravel()})
df_pred_test
# -

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(y_test, label='Truth', marker='o')
ax.plot(df_pred_test, label='Neural Network Predictions', marker='x')
ax.legend()
plt.show()
print('type of y_test is ', type(y_test))
print('type of df_pred_test is ', type(df_pred_test))

forecast_accuracy(df_pred_test["predictions"].to_numpy(), y_test)

# +
#{'mape': 0.42619692749642285,
# 'me': -228466.9305391664,
# 'mae': 753582.7795691834,
# 'mpe': 0.1954761148314077,
# 'mse': 946360530436.5908,
# 'rmse': 972810.6344179173}
