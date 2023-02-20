# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 00:19:24 2019

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from preprocess import fillEmpty
from filteration import removeConstantFeature
from filteration import removeQuasiConstant
from filteration import removeDuplicateFeature
from filteration import removeCorelatedFeature
#from filteration import removeUnivariate
from embedded import removeLasso
from embedded import removeRidge
from knn import accuracyKnn
from knn import accuracyDecision

dataset = pd.read_csv('dataset_5_arrhythmia.csv')
acc = []
acc1 = []
label = ['raw', 'Constant', 'Quasi', 'Duplicate', 'Correlated', 'Lasso', 'Ridge']

#preprocess
uniVal = 10
data = fillEmpty(dataset,uniVal)

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['class'], axis=1),
    data['class'],
    test_size=0.3,
    random_state=0)
print(X_train.shape)

#accuracy on raw data
acc.append(accuracyKnn(X_train,X_test,y_train,y_test))
acc1.append(accuracyDecision(X_train,X_test,y_train,y_test))

#Remove Constant Features
X_train, X_test = removeConstantFeature(X_train,X_test)
print(X_train.shape)
acc.append(accuracyKnn(X_train,X_test,y_train,y_test))
acc1.append(accuracyDecision(X_train,X_test,y_train,y_test))

#Remove Quasi Constant
varianceAllowed = 0.01
X_train_quasi, X_test_quasi = removeQuasiConstant(X_train, X_test, varianceAllowed)
print(X_train.shape)
acc.append(accuracyKnn(X_train_quasi,X_test_quasi,y_train,y_test))
acc1.append(accuracyDecision(X_train_quasi,X_test_quasi,y_train,y_test))

#Remove Duplicate Feature
X_train_dup, X_test_dup = removeDuplicateFeature(X_train, X_test)
print(X_train.shape)
acc.append(accuracyKnn(X_train_dup,X_test_dup,y_train,y_test))
acc1.append(accuracyDecision(X_train_dup,X_test_dup,y_train,y_test))

#Remove Correlated Feature
threshold = 0.8
X_train_corr, X_test_corr = removeCorelatedFeature(X_train, X_test, threshold)
acc.append(accuracyKnn(X_train_corr,X_test_corr,y_train,y_test))
acc1.append(accuracyDecision(X_train_corr,X_test_corr,y_train,y_test))

#Remove Unvariate roc-auc
#X_train, X_test, y_train, y_test = removeUnivariate(X_train, X_test, y_train, y_test)

#lasso
X_train_lasso, X_test_lasso = removeLasso(X_train, X_test, y_train, y_test)
acc.append(accuracyKnn(X_train_lasso,X_test_lasso,y_train,y_test))
acc1.append(accuracyDecision(X_train_lasso,X_test_lasso,y_train,y_test))

#ridge
acc.append(removeRidge(X_train,X_test, y_train, y_test))

plt.plot(label,acc)
plt.plot(label,acc1)
