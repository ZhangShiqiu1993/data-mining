# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# y = dataset.loc[:, 'Profit'].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
label_encoder = LabelEncoder()
X[:, 3] = label_encoder.fit_transform(X[:, 3])
one_hot_encoder = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder.fit_transform(X).toarray()

X = X[:, 1:]  # avoid dummy variable trap

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# build the optimal model using backward elimination
import statsmodels.formula.api as sm

#X = np.append(arr=X, values=np.ones((np.size(X, 0), 1)).astype(int), axis=1)
# add b_0 column at the begin instead of the end
X = np.append(arr=np.ones((np.size(X, 0), 1)).astype(int), values=X, axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# x2 has the highest p value, then we remove 2 from next model

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
