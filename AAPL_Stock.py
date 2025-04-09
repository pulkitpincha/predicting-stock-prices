# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 03:23:46 2023

@author: stimp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM    
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#loading the data
df = pd.read_csv("Datasets/AAPL.csv")
df.head()

df = df.drop(['date', 'raw_close', 'change_percent', 'avg_vol_20d', 'Month'], axis = 1)
df.head()

plt.plot(df.close)

ma100 = df.close.rolling(100).mean()

plt.figure(figsize = (12,6))
plt.plot(df.close)
plt.plot(ma100, 'r')

ma200 = df.close.rolling(200).mean()

plt.figure(figsize = (12,6))    
plt.plot(df.close)
plt.plot(ma100, 'r', label = '100 Day MA')
plt.plot(ma200, 'g', label = '200 Day MA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

data_training = pd.DataFrame(df[0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df[int(len(df)*0.7):int(len(df))])
data_training = data_training.drop(['open', 'high', 'low', 'Interest', 'volume'], axis = 1)
data_testing = data_testing.drop(['open', 'high', 'low', 'Interest', 'volume'], axis = 1)

print(data_training.shape)
print(data_testing.shape)

scaler = MinMaxScaler()

data_training_array = scaler.fit_transform(data_training)
data_training_array

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
               input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)

data_testing.head()
data_testing.tail(100)
past_100_days = data_testing.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
final_df.head()

input_data = scaler.fit_transform(final_df)
input_data
scaler.scale_
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)

y_predicted = model.predict(x_test)
y_predicted.shape

y_test

scaler.scale_
sc = scaler.scale_[0]
scale_factor = 1/sc
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()