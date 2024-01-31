# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:45:44 2024

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

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerAutoencoder, self).__init__()

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_encoder_layers
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_decoder_layers
        )
        
        self.fc1 = nn.Linear(input_size, d_model)
        self.fc2 = nn.Linear(d_model, input_size)
        
    def forward(self, x):
        x = self.fc1(x)  # Project the input to the model dimension
        
        # Use the transformer encoder for encoding
        encoding = self.transformer_encoder(x)
        
        # Aggregate information across the sequence dimension (mean pooling)
        encoding = encoding.mean(dim=1)
        
        # Expand the dimensions to match the expected input size for the decoder
        encoding = encoding.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Use the transformer decoder for decoding
        decoding = self.transformer_decoder(x, encoding)
        
        return self.fc2(decoding)
    
loss_function = SoftDTWLossPyTorch(gamma = .1)
n_windows = 100
#X, labels = generate_batches(n_windows, 700, p = [.5, .1, .3, .05, .05])
#X1, X2, sizes, X, labels = deinterleave(X,labels, impurity = .4)
X,labels = generate_impure_signals(n_windows, ws = [300,350,400,450, 500,550,600,650, 700])
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

input_size = 3
model_dimension = 64
nheads = 16
encoder_layers = 3
decoder_layers = 3

model = TransformerAutoencoder(input_size, model_dimension, nheads, encoder_layers, decoder_layers)

optimizer = torch.optim.Adam(model.parameters(),
                                 lr = 1e-4,
                                 weight_decay = 1e-7)
for epoch in range(epochs):
    losses = []
    if epoch%10==1:
        print(avg_losses[-1])
    for i,x in enumerate(X):
        seq_len = x.shape[0]
        x = torch.tensor(np.array(x, dtype="float32"), dtype=torch.float32).reshape(1, seq_len, 3)
        #x_reshaped = x.reshape(-1, 4*largest_window)
        reconstructed = model(x)
        
        loss = loss_function(reconstructed, x)
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_losses.append(np.mean(losses))
    
plt.plot(avg_losses)
X_encoded = []
model.eval()
for xt in X:
    seq_len = xt.shape[0]
    xt = torch.tensor(np.array(xt, dtype="float32"), dtype=torch.float32).reshape(1, seq_len, 3)
    x_projected = model.fc1(xt)
    x_encoded = model.transformer_encoder(x_projected).mean(dim=1).detach().numpy()
    X_encoded.append(x_encoded[0])
X_encoded = np.array(X_encoded)

from sklearn.cluster import SpectralClustering
labels_pred = SpectralClustering(n_clusters=5, affinity = "nearest_neighbors").fit_predict(X_encoded)
print(homogeneity_completeness_v_measure(labels, labels_pred))