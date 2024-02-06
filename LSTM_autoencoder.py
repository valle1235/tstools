# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:12:00 2024

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

class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim=16, n_layers = 1):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.n_layers = n_layers
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=self.n_layers,
            batch_first=True
        )

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.reshape((batch_size, seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((batch_size, self.n_layers*self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, input_dim=16, n_features=4, n_layers = 1):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.n_layers = n_layers
        self.rnn1 = nn.LSTM(
            input_size=input_dim*self.n_layers,
            hidden_size=input_dim*self.n_layers,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim*self.n_layers,
            hidden_size=self.hidden_dim*self.n_layers,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim*self.n_layers, n_features)

    def forward(self, x, seq_len):
        batch_size = x.shape[0]
        x = x.repeat(seq_len, 1)
        x = x.reshape((batch_size, seq_len, self.n_layers*self.input_dim))
        
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((batch_size, seq_len, self.hidden_dim*self.n_layers))
        return self.output_layer(x)


class LSTMAE(nn.Module):
    def __init__(self, n_features, embedding_dim=16, device='cuda', batch_size=1, n_layers = 1):
        super(LSTMAE, self).__init__()
        self.encoder = Encoder(n_features = n_features, embedding_dim = embedding_dim, n_layers = n_layers)
        self.decoder = Decoder(input_dim = embedding_dim, n_features = n_features, n_layers = n_layers)

    def forward(self, x, seq_len):
        x = self.encoder(x)
        x = self.decoder(x, seq_len)
        return x

loss_function = SoftDTWLossPyTorch(gamma = .1)
n_windows = 1000
X, labels = generate_batches(n_windows, 700, p = [.5, .1, .3, .05, .05])
X1, X2, sizes, X, labels,_ = deinterleave(X,labels, impurity = .4)
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

avg_losses = []
epochs = 50
model = LSTMAE(4, n_layers = 1)
optimizer = torch.optim.Adam(model.parameters(),
                                 lr = 1e-4,
                                 weight_decay = 1e-7)
for epoch in range(epochs):
    losses = []
    if epoch%10==1:
        print(avg_losses[-1])
    for i,x in enumerate(X):
        seq_len = x.shape[0]
        x = torch.tensor(np.array(x, dtype="float32"), dtype=torch.float32).reshape(1, seq_len, 4)
        #x_reshaped = x.reshape(-1, 4*largest_window)
        reconstructed = model(x, seq_len)
        
        loss = loss_function(reconstructed, x)
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_losses.append(np.mean(losses))
    
plt.plot(avg_losses)
X_encoded = []
rf_means = np.array([0,.1,.2,.25,.5])
amp_scalings = np.array([.1,.6,.9,.8,.2])
amp_freq = np.array([1/15, 1/35, 1/55, 1/45, 1/25])
aoa_start = np.array([50,49,48.5,47,46])
aoa_coef = np.array([10**(-5.1),10**(-3.8),10**(-4.6),10**(-4.1),10**(-3.5)]),
pw_start = np.array([1, .9, 1.4, 1.1, .8])
X, labels = generate_batches(100, 700, p = [.5, .1, .3, .05, .05], rf_means=rf_means, amp_scalings=amp_scalings,
                             amp_freq=amp_freq, aoa_start=aoa_start, aoa_coef=aoa_coef, pw_start=pw_start)
X1, X2, sizes, X, labels,_ = deinterleave(X,labels, impurity = .4)

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

model.eval()
for xt in X:
    seq_len = xt.shape[0]
    xt = torch.tensor(np.array(xt, dtype="float32"), dtype=torch.float32).reshape(1, seq_len, 4)
    x_encoded = model.encoder(xt).detach().numpy()
    X_encoded.append(x_encoded[0])
X_encoded = np.array(X_encoded)

from sklearn.cluster import SpectralClustering
labels_pred = SpectralClustering(n_clusters=5, affinity = "nearest_neighbors").fit_predict(X_encoded)
print(homogeneity_completeness_v_measure(labels, labels_pred))