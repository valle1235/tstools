# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 05:59:46 2024

@author: Valdemar
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from dtw import dtw
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.stats import entropy
from statsmodels.tsa.statespace.varmax import VARMAX
#from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima, ARIMA
import bz2
import torch.nn as nn
from statsmodels.tsa.arima_process import ArmaProcess
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.decomposition import PCA

import torch
from torchvision import datasets
from torchvision import transforms
import stumpy
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import quantile_transform
from sklearn.metrics import homogeneity_completeness_v_measure

from generate_ts import generate, generate_impure_signals,generate_pdw, generate_batches
from deinterleave import deinterleave
import umap.umap_ as umap
from matrixprofile.algorithms import stomp
from tslearn.metrics.soft_dtw_loss_pytorch import SoftDTWLossPyTorch


def score(labels_pred, labels, label_order):
    c=0
    temp_labels = np.copy(labels_pred)
    seen = []
    for label in temp_labels:
        if label in seen:
            continue
        idx = np.where(temp_labels == label)
        labels_pred[idx] = label_order[c]
        seen.append(label)
        c+=1
        
    print(f"V measure: {homogeneity_completeness_v_measure(labels, labels_pred)}")
    
    return homogeneity_completeness_v_measure(labels, labels_pred), labels_pred

class AE(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4*input_size, 512),
            
            torch.nn.Conv1d(1, 6, 7),
            torch.nn.SiLU(),
            torch.nn.Conv1d(6, 12, 5),
            torch.nn.SiLU(),
            torch.nn.Conv1d(12, 6, 5),
            torch.nn.SiLU(),
            torch.nn.Conv1d(6, 1, 3),
            torch.nn.SiLU(),
            
            torch.nn.Linear(512 - 16, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 6),
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(6, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.SiLU(),
            
            torch.nn.Linear(1024, 4*input_size),
            torch.nn.Sigmoid()
            
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

loss_function = SoftDTWLossPyTorch(gamma = .1)
n_windows = 100
X, labels = generate_batches(n_windows, 700, p = [.5, .1, .3, .05, .05])
X1, X2, sizes, X, labels = deinterleave(X,labels, impurity = .4)
n_windows = 5*n_windows

X_complete = np.array([])
for x in X:
    if X_complete.shape[0] == 0:
        X_complete = x
        continue
    X_complete = np.append(X_complete, x, axis = 0)
X_complete = quantile_transform(X_complete, n_quantiles = 1000)
transformed_X = []
t = 0
for x in X:
    transformed_X.append(X_complete[t:t+x.shape[0]])
    t+=x.shape[0]
X = transformed_X


largest_window = 0
for x in X:
    if x.shape[0] > largest_window:
        largest_window = x.shape[0]
print(largest_window)
X_ae = np.zeros((n_windows, largest_window, 4))
for i,x in enumerate(X):
    padding = largest_window-x.shape[0]
    x = np.concatenate((x, np.zeros((padding, x.shape[1]))))
    X_ae[i] = x
X_ae = np.array(X_ae)
avg_losses = []
epochs = 100
model = AE(largest_window)
optimizer = torch.optim.Adam(model.parameters(),
                                 lr = 1e-4,
                                 weight_decay = 1e-8)
for epoch in range(epochs):
    losses = []
    if epoch%10==1:
        print(avg_losses[-1])
    for i,x in enumerate(X_ae):
        x = torch.tensor(x, dtype=torch.float32)
        x_reshaped = x.reshape(-1, 4*largest_window)
        reconstructed = model(x_reshaped).reshape(1,largest_window,4)
        
        loss = loss_function(reconstructed, x.reshape(1,largest_window,4))
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_losses.append(np.mean(losses))
plt.plot(avg_losses)
X_encoded = []
for xt in X_ae:
    xt = torch.tensor(xt, dtype=torch.float32)
    xt = xt.reshape(-1, 4*largest_window)
    x_encoded = model.encoder(xt).detach().numpy()
    X_encoded.append(x_encoded[0])
X_encoded = np.array(X_encoded)

'''
reducer = umap.UMAP()
encoded = []

for i,x in enumerate(X):
    print(i)
    reducer.fit(x)
    encoded.append(reducer.transform(x.reshape(1,-1)))
'''
X2 = []
for x in X:
    X2.append(np.mean(x[:,0:2], axis=0))
X2 = np.array(X2)
from sklearn.cluster import SpectralClustering
        
labels_pred = SpectralClustering(n_clusters=5, affinity = "nearest_neighbors").fit_predict(X_encoded)
print(homogeneity_completeness_v_measure(labels, labels_pred))
    