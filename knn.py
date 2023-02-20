# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:02:15 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def accuracyKnn(X_train,X_test,y_train,y_test):
    classifierknn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifierknn.fit(X_train,y_train)
    y_predknn = classifierknn.predict(X_test)
    accuracy = accuracy_score(y_test, y_predknn)
    return accuracy

def accuracyDecision(X_train,X_test,y_train,y_test):
    classifierdt = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifierdt.fit(X_train,y_train)
    y_predsvm = classifierdt.predict(X_test)
    accuracy1 = accuracy_score(y_test, y_predsvm)
    return accuracy1
