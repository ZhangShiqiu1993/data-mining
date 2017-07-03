# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
training_set = scaler.fit_transform(training_set)

X_train = training_set[0:1257]
y_train = training_set[1:1258]

X_train = np.reshape(X_train, (1257, 1, 1))

from keras.models import Sequential
from keras.layers import Dense, LSTM
regressor = Sequential()
regressor.add(LSTM(units=4, activation='sigmoid', input_shape = (None, 1)))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

inputs = real_stock_price
inputs = scaler.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))

def visualize(real_data, predicted_data):
    plt.plot(real_data, color = 'red', label = 'Real Google Stock Price')
    plt.plot(predicted_data, color = 'blue', label = 'Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()

def predict_normalize(input_data):
    predicated_data = regressor.predict(input_data)
    predicated_data = scaler.inverse_transform(predicated_data)
    return predicated_data
    
predicted_stock_price = predict_normalize(inputs)
visualize(real_stock_price, predicted_stock_price)

real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train = real_stock_price_train.iloc[:,1:2].values
predicted_stock_price_train = predict_normalize(X_train)
visualize(real_stock_price_train, predicted_stock_price_train)

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
