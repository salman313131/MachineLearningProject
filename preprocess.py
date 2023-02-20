# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 00:21:02 2019

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fillEmpty(data, uniVal):
    categorical = []

    del data['J']

    for i in range(0,data.shape[1]):
        temp = []
        temp = np.unique(data[data.columns[i]])
        if len(temp) <= uniVal:
            categorical.append(data.columns[i])
    continuous = []
    continuous  = list(set(data.columns) - set(categorical))
    
    for j in continuous:
        data[j] = data[j].fillna((data[j].mean()))
        
    for k in categorical:
        data[k] = data[k].fillna((data[k].mode()[0]))
    
    return data
