
"""
Created on Tue Apr  2 17:16:59 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def removeLasso(X_train,X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1'))
    sel_.fit(scaler.transform(X_train), y_train)
    
    X_train_lasso = pd.DataFrame(sel_.transform(X_train))
    X_test_lasso = pd.DataFrame(sel_.transform(X_test))
    
    X_train_lasso.columns = X_train.columns[(sel_.get_support())]
    X_test_lasso.columns = X_train.columns[(sel_.get_support())]
    return X_train_lasso, X_test_lasso

def removeRidge(X_train,X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    l2 =  LogisticRegression(C=1, penalty = 'l2')
    l2.fit(X_train, y_train)
    y_predict = l2.predict(X_test)
    ridgecm = confusion_matrix(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    return accuracy


