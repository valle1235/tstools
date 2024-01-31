# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 05:45:49 2024

@author: Valdemar
"""

import numpy as np
import stumpy
from sklearn.cluster import AgglomerativeClustering
from time_series_clustering_stuff import score
from generate_ts import generate, generate_impure_signals, generate_pdw
from time import time
from numba import cuda
from sklearn.metrics import homogeneity_completeness_v_measure
from matrixprofile.algorithms import mpdist
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN

all_gpu_devices = [device.id for device in cuda.list_devices()]
times = []
window_sizes = np.array([400, 430, 450, 480, 500])

n_windows = 100
X, labels = generate_pdw(n_windows, impurity = 0.1, ws=window_sizes)

labels = np.array(labels)
label_order = []
for l in labels:
    if l in label_order:
        continue
    label_order.append(l)

mp_distance = np.zeros((n_windows, n_windows))
st = time()
for i,x1 in enumerate(X):
    for j,x2 in enumerate(X):
        if i == j:
            mp_distance[i,j] =0
            continue
        dists = 0
        for dim in range(x1.shape[1]):
            dists += (mpdist(x1[:,dim], x2[:,dim], 15, n_jobs = -1, threshold = .1))
        mp_distance[i,j] = (dists)
mp_labels = AgglomerativeClustering(n_clusters=5, metric="precomputed", linkage="complete").fit_predict(mp_distance)

v,_ = score(mp_labels, labels, label_order)
print(time()-st)