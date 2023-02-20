# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 00:31:09 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

def removeConstantFeature(X_train,X_test):
    constant_features = [
        feat for feat in X_train.columns if X_train[feat].std() == 0
    ]
    X_train.drop(labels=constant_features, axis=1, inplace=True)
    X_test.drop(labels=constant_features, axis=1, inplace=True)
    return X_train,X_test
    
def removeQuasiConstant(X_train, X_test, varianceAllowed):
    sel = VarianceThreshold(threshold=varianceAllowed)
    sel.fit(X_train)
    features_to_keep = X_train.columns[sel.get_support()]
    X_train = sel.transform(X_train)
    X_test = sel.transform(X_test)
    X_train= pd.DataFrame(X_train)
    X_train.columns = features_to_keep
    X_test= pd.DataFrame(X_test)
    X_test.columns = features_to_keep
    return X_train, X_test
    
def removeDuplicateFeature(X_train, X_test):
    duplicated_feat = []
    for i in range(0, len(X_train.columns)):
        if i % 10 == 0:  # this helps me understand how the loop is going
            print(i)
    
        col_1 = X_train.columns[i]
    
        for col_2 in X_train.columns[i + 1:]:
            if X_train[col_1].equals(X_train[col_2]):
                duplicated_feat.append(col_2)
                
    len(duplicated_feat)
    X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
    X_test.drop(labels=duplicated_feat, axis=1, inplace=True)
    return X_train, X_test

def removeCorelatedFeature(X_train, X_test, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = X_train.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    col_corr
    X_train_corr = X_train.copy()
    X_test_corr = X_test.copy()
    X_train_corr.drop(labels=col_corr, axis=1, inplace=True)
    X_test_corr.drop(labels=col_corr, axis=1, inplace=True)
    return X_train_corr, X_test_corr

'''
def removeUnivariate(X_train, X_test, y_train, y_test):
    roc_values = []
    for feature in X_train.columns:
        clf = DecisionTreeClassifier()
        clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
        y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
        roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))
    roc_values = pd.Series(roc_values)
    roc_values.index = X_train.columns
    roc_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))
    selected_feat = roc_values[roc_values>0.5]
    print(len(selected_feat), X_train.shape[1])
    return X_train, X_test, y_train, y_test
'''