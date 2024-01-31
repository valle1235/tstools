# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:55:12 2024

@author: Valdemar
"""

#Basic stuff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import quantile_transform
from sklearn.metrics import homogeneity_completeness_v_measure

#Autoencoder stuff
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from tslearn.metrics.soft_dtw_loss_pytorch import SoftDTWLossPyTorch

#Dataset stuff
from generate_ts import generate, generate_impure_signals,generate_pdw, generate_batches
from deinterleave import deinterleave
from torch.nn.functional import pairwise_distance

class LSTM_Siamese(nn.Module):
    def __init__(self, n_features, embedding_dim=2, n_layers = 1):
        super(LSTM_Siamese, self).__init__()
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

    def forward_one_val(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.reshape((batch_size, seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        emb = hidden_n.reshape((batch_size, self.n_layers*self.embedding_dim))
        return emb
    
    def forward(self, x1, x2):
        emb1 = self.forward_one_val(x1)
        emb2 = self.forward_one_val(x2)
        
        return emb1, emb2
    
def contrastive_loss(embedding1, embedding2, margin=1.0):
    distance = pairwise_distance(embedding1, embedding2)
    loss = torch.mean((1 - distance) ** 2)  # Contrastive loss
    return loss

loss_function = SoftDTWLossPyTorch(gamma = .1)
n_windows = 100
X, labels = generate_batches(n_windows, 700, p = [.5, .1, .3, .05, .05])
X1, X2, sizes, X, labels, wid = deinterleave(X,labels, impurity = .4)
#X,labels = generate_impure_signals(n_windows, ws = [300,350,400,450, 500,550,600,650, 700])
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
epochs = 200
avg_losses = []

model = LSTM_Siamese(4)
optimizer = torch.optim.Adam(model.parameters(),
                                 lr = 1e-4,
                                 weight_decay = 1e-7)

for epoch in range(epochs):
    k=0
    window = wid[0]
    losses = []
    if epoch%10 == 1:
        print(avg_losses[-1])
    for i in range(len(X)):
        if i not in window:
            k+=1
            window = wid[k]
        for j in window:
            if j == i:  continue
            seq_len1 = X[i].shape[0]
            seq_len2 = X[j].shape[0]
            
            x1 = torch.tensor(np.array(X[i], dtype="float32"), dtype=torch.float32).reshape(1, seq_len1, 4)
            x2 = torch.tensor(np.array(X[j], dtype="float32"), dtype=torch.float32).reshape(1, seq_len2, 4)
            
            emb1, emb2 = model(x1, x2)
            loss = contrastive_loss(emb1, emb2)
               
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    avg_losses.append(np.mean(losses))
    
plt.plot(avg_losses)
plt.show()
X_encoded = []
model.eval()
for xt in X:
    seq_len = xt.shape[0]
    xt = torch.tensor(np.array(xt, dtype="float32"), dtype=torch.float32).reshape(1, seq_len, 4)
    x_encoded = model.forward_one_val(xt).detach().numpy()
    X_encoded.append(x_encoded[0])
X_encoded = np.array(X_encoded)

from sklearn.cluster import SpectralClustering, KMeans
labels_pred = KMeans(n_clusters=5).fit_predict(X_encoded)
print(homogeneity_completeness_v_measure(labels, labels_pred))
plt.scatter(X_encoded[:,0], X_encoded[:,1], c = labels)