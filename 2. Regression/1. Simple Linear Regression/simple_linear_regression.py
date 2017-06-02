# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

def visulize_result(X, y, title, xlabel, ylabel):
    plt.scatter(X_train, y_train, color='red', marker='x')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title()
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.show()

def simple_linear_regression_visulization(X, y):
    return visulize_result(X, y, "Salary vs Experience", "Years of Experience", "Salary")

simple_linear_regression_visulization(X_train, y_train)
simple_linear_regression_visulization(X_test, y_test)