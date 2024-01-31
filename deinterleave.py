# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:36:55 2024

@author: Valdemar
"""

import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
from generate_ts import generate, generate_impure_signals,generate_pdw,generate_batches

def multi_corr(ts1,ts2):
    ret = 0
    for dim in range(ts1.shape[1]):
        ret += max(np.correlate(ts1[:,dim], ts2[:,dim], mode = 'full'))
    return ret
def corr_dist(ts1, ts2):
    corrt1t2 = multi_corr(ts1, ts2)
    corrt1 = multi_corr(ts1,ts1)
    corrt2 = multi_corr(ts2,ts2)
    return (1 - corrt1t2/(corrt1*corrt2))

def deinterleave(X, labels, impurity = .05):
    X_deinterleaved = []
    labels_batched = []
    sizes = []
    labels_window_wize = []
    X_deinterleaved_as_ts = []
    
    X = np.array(X, dtype="object")
    
    for label, x in zip(labels, X):
        curr_batch_x = np.array([])
        curr_batch_labels = []
        curr_sizes = []
        
        for l in set(label):
            idx = np.where(label == l)[0].tolist()
            idx_conjugate = np.where(label != l)[0]
            for _ in range(int(len(idx)*impurity)):
                idx.append(np.random.choice(idx_conjugate, 1)[0])
            idx = sorted(idx)
            temp = x[idx]
            curr_batch_labels.append(l)
            curr_sizes.append(temp.shape[0])
            X_deinterleaved_as_ts.append(temp)
            labels_window_wize.append(l)
            if curr_batch_x.shape[0] == 0:
                curr_batch_x = temp
                continue
            curr_batch_x = np.concatenate((curr_batch_x, temp))
            
        X_deinterleaved.append(np.array(curr_batch_x))
        labels_batched.append(np.array(curr_batch_labels,dtype="int"))
        sizes.append(curr_sizes)
    return X_deinterleaved, labels_batched, sizes, X_deinterleaved_as_ts, labels_window_wize

'''
X, labels = generate_batches(30, 500)
plt.scatter(np.arange(500),X[0][:,1], c = labels[0])
plt.show()
X, labels, sizes, X_ts, l_wz = deinterleave(X,labels)

nu = 1
dist_function = lambda x,y: (np.abs(np.dot(x-y,x-y)))**nu
labels_pred = [[0,1,2,3,4]]

for i in range(len(sizes)-1):
    temp_labels = []
    for j in range(len(sizes[i+1])):
        st2 = sizes[i+1][j]
        end2 = -1
        if j != len(sizes[i+1]) -1:
            end2 = sizes[i+1][j+1]
        
        dists = []
        for k in range(len(sizes[i])):
            st1 = sizes[i][k]
            end1 = -1
            if k != len(sizes[i]) -1:
                end1 = sizes[i][k+1]
            
            if end1 != -1:
                ts1 = X[i][st1:st1+end1]
            else:
                ts1 = X[i][st1:]
            if end2 != -1:
                ts2 = X[i+1][st2:st2+end2]
            else:
                ts2 = X[i+1][st2:]
            d = corr_dist(ts1, ts2)
            dists.append(d)
        
        min_dist = np.argmin(dists)
        temp_labels.append(labels_pred[i][min_dist])
    labels_pred.append(temp_labels)
'''