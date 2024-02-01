# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:19:32 2024

@author: Valdemar
"""

import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure
from deinterleave import deinterleave
import matplotlib.pyplot as plt
from sklearn.preprocessing import quantile_transform

def evaluate(X,labels, method, impurities = np.arange(0,1,.05)):
    v_scores = []
    
    for impurity in impurities:
        X1, X2, sizes, X_deinterleaved, labels_deinterleaved, wid = deinterleave(X, labels, impurity = impurity)
        
        X_complete = np.array([])
        for x in X_deinterleaved:
            if X_complete.shape[0] == 0:
                X_complete = x
                continue
            X_complete = np.append(X_complete, x, axis = 0)
        X_complete = quantile_transform(X_complete, n_quantiles = 1000)
        transformed_X = []
        t = 0
        for x in X_deinterleaved:
            transformed_X.append(X_complete[t:t+x.shape[0]])
            t+=x.shape[0]
        X_deinterleaved = transformed_X
        
        labels_pred = method(X_deinterleaved, wid)
        score = homogeneity_completeness_v_measure(labels_deinterleaved, labels_pred)
        
        v_scores.append(score[2])
        print(v_scores[-1])
    
    plt.plot(impurities, v_scores)
    plt.title("V score vs impurity level")
    plt.xlabel("Impurity")
    plt.ylabel("V score")
    plt.grid()
    plt.show()
    
    return v_scores