# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_classifier(optimizer = 'adam'):
    classifier = Sequential()
    classifier.add(Dense(kernel_initializer='uniform', activation = 'relu', input_dim = 11, units=6))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(kernel_initializer = 'uniform', activation = 'relu', units=6))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(kernel_initializer = 'uniform', activation = 'sigmoid', units=1))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def run_K_fold_cross_validation(k=10):
    classifier = KerasClassifier(build_fn=build_classifier, batch_size=16, epochs=128)
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
    mean = accuracies.mean()
    variance = accuracies.std()
    print("mean : " + mean)
    print("std : " + variance)

def run_grid_search():
    classifier = KerasClassifier(build_fn=build_classifier)
    parameters = {'batch_size' : [16, 25, 32],
                  'epochs': [32, 64, 128, 256, 512],
                  'optimizer' : ['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator=classifier, 
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=10)
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print("best parameters : " + best_parameters)
    print("best accuracy : " + best_accuracy)
    
run_grid_search()