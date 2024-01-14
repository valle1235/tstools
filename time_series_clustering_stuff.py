import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from dtw import dtw
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.stats import entropy
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import bz2
import torch.nn as nn
from statsmodels.tsa.arima_process import ArmaProcess
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.decomposition import PCA

import torch
from torchvision import datasets
from torchvision import transforms

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(3*100, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3*100),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def JSD(model1, model2,N=50000):
    sample1,_ = model1.sample(N)
    #sample2,_ = model2.sample(N)
    
    P = np.exp(model1.score_samples(sample1))
    Q = np.exp(model2.score_samples(sample1))
    
    idx = np.where(P < 1e-300)
    P[idx] = 1e-300
    idx = np.where(Q < 1e-300)
    Q[idx] = 1e-300
    M = 0.5 * (P + Q)

    return 0.5 * (entropy(P, M) + entropy(Q, M))

def f1(w):
    ar1 = [.1, -.05, .0007, -.05]
    ma1 = [1]
    
    ar2 = [1, -.9]
    ma2 = [1]
    
    ar3 = [1, -.86]
    ma3 = [1]
    
    AR_object1 = ArmaProcess(ar1, ma1)
    x = AR_object1.generate_sample(nsample=w)
    
    AR_object2 = ArmaProcess(ar2, ma2)
    y = AR_object2.generate_sample(nsample=w)
    
    AR_object3 = ArmaProcess(ar3, ma3)
    z = AR_object3.generate_sample(nsample=w)
        
    return np.array([x,y,z]).T

def f2(w):
    ar1 = [.1, -.1, .0007, .01]
    ma1 = [1]
    
    ar2 = [1, -.3, .1, -.16, .9]
    ma2 = [1]
    
    ar3 = [1, -.9]
    ma3 = [1]
    
    AR_object1 = ArmaProcess(ar1, ma1)
    x = AR_object1.generate_sample(nsample=w)
    
    AR_object2 = ArmaProcess(ar2, ma2)
    y = AR_object2.generate_sample(nsample=w)
    
    AR_object3 = ArmaProcess(ar3, ma3)
    z = AR_object3.generate_sample(nsample=w)
        
    return np.array([x,y,z]).T

def f3(w):
    ar1 = [1, -.9, -.1, .196, -.4, .2]
    ma1 = [1]
    
    ar2 = [-1, -.9, .1, -.00000001, .0000001]
    ma2 = [1, .5, .5]
    
    ar3 = [1, -.99]
    ma3 = [1]
    
    AR_object1 = ArmaProcess(ar1, ma1)
    x = AR_object1.generate_sample(nsample=w)
    
    AR_object2 = ArmaProcess(ar2, ma2)
    y = AR_object2.generate_sample(nsample=w)
    
    AR_object3 = ArmaProcess(ar3, ma3)
    z = AR_object3.generate_sample(nsample=w)
        
    return np.array([x,y,z]).T

def f4(w):
    ar1 = [1, -.9, -.01, .0196, -.04, .2]
    ma1 = [1]
    
    ar2 = [-1, .9, -.3, -.1, .0000001]
    ma2 = [1, .5, .1]
    
    ar3 = [1, -.99, .1, -.1]
    ma3 = [1]
    
    AR_object1 = ArmaProcess(ar1, ma1)
    x = AR_object1.generate_sample(nsample=w)
    
    AR_object2 = ArmaProcess(ar2, ma2)
    y = AR_object2.generate_sample(nsample=w)
    
    AR_object3 = ArmaProcess(ar3, ma3)
    z = AR_object3.generate_sample(nsample=w)
        
    return np.array([x,y,z]).T

def f5(w):
    ar1 = [1, -.9]
    ma1 = [1]
    
    ar2 = [-1, -.9]
    ma2 = [1, .5223, -.545]
    
    ar3 = [1, -.9]
    ma3 = [1]
    
    AR_object1 = ArmaProcess(ar1, ma1)
    x = AR_object1.generate_sample(nsample=w)
    
    AR_object2 = ArmaProcess(ar2, ma2)
    y = AR_object2.generate_sample(nsample=w)
    
    AR_object3 = ArmaProcess(ar3, ma3)
    z = AR_object3.generate_sample(nsample=w)
        
    return np.array([x,y,z]).T

def random_walk(p, scaling = .1):
    moveset = np.arange(p.shape[1]) - p.shape[1]//2
    moveset[p.shape[1]//2:] += 1
    
    walk = np.zeros(p.shape[0])
    for i, q in enumerate(p):
        move = moveset[np.argmax(q)]
        if i == 0:
            walk[i] = move
            continue
        walk[i] = walk[i-1] + move
    
    return scaling*walk/np.max(np.abs(walk))
    

def gaussian_distance(mu1, mu2, sigma1, sigma2):
    s = (sigma1+sigma2)/2
    mu = mu1 - mu2
    D1 = np.dot(np.linalg.inv(s), mu)
    D1 = np.dot(mu.T, D1)/8
    D2 = .5*np.log(np.linalg.det(s)/(np.sqrt(np.linalg.det(sigma1)*np.linalg.det(sigma2))))
    return D1 + D2

def HMM_distance(hmm1, hmm2):
    emission_distance = 0
    mu1s = hmm1.means_
    covs1 = hmm1.covars_
    mu2s = hmm2.means_
    covs2 = hmm2.covars_
    
    for i in range(mu1s.shape[0]):
        emission_distance += gaussian_distance(mu1s[i], mu2s[i], covs1[i], covs2[i])
        
    trans1 = hmm1.transmat_
    trans2 = hmm2.transmat_
    trans_dist = 0
    
    for a,b in zip(trans1, trans2):
        trans_dist += np.sqrt(np.abs(np.dot(a,b)))
    
    return trans_dist + emission_distance

def plot_series(ts, labels, length, dim):
    col = ['red', 'blue', 'green', 'yellow', 'purple']
    t = 0
    k=0
    for s,l in zip(ts,labels):
        if k == length:
            break
        plt.plot(np.arange(t,s.shape[0]+t), s[:,dim], c = col[l])
        t += s.shape[0]
        k+=1
    plt.show()
    
def calc_euclid(ts1, ts2):
    td = max(ts1.shape[0] - ts2.shape[0], ts2.shape[0] - ts1.shape[0])
    if ts1.shape[0] < ts2.shape[0]:
        mins = ts1
        maxs = ts2
    else:
        mins = ts2
        maxs = ts1
    dists = []
    for k in range(td+1):
        temp = 0
        for i in range(mins.shape[0]):
            temp += np.dot(mins[i] - maxs[i+k], mins[i] - maxs[i+k])/mins.shape[0]
        dists.append(temp)
    return np.mean(dists)

def GMM_distance(X):
    distance_matrix = np.zeros((n_windows,n_windows))
    models = []

    for x in X:
        model = GaussianMixture(n_components=3).fit(x)
        models.append(model)
        
    for i, ts1 in enumerate(models):
        for j,ts2 in enumerate(models):
            d = JSD(ts1, ts2)
            if i == j:
                d = 0
            distance_matrix[i,j] = d
    return distance_matrix

def ARIMA_error(X):
    datapoints = []

    for x in X:
        model1 = auto_arima((4,1,2))
        model1.fit(x[:,0])
        model2 = auto_arima((4,1,2))
        model2.fit(x[:,1])
        model3 = auto_arima((4,1,2))
        model3.fit(x[:,2])
        input(model1.params())
        datapoints.append(np.array([model1.params(), model2.params(), model3.params()]))
        
    return np.array(datapoints)

def transform_ts_to_datapoints(X):
    datapoints = []
    dim1 = X[0].shape[1]
    for x in X:
        datapoint = []
        for i in range(dim1):
            for j in range(dim1):
                corr = np.correlate(x[:,j], x[:,i])/x.shape[0]
                datapoint.append(np.max(corr))
        datapoint.extend(np.mean(x, axis=0).tolist())
        datapoint.extend(np.var(x, axis=0).tolist())
        
        model1 = auto_arima((4,1,2))
        model1.fit(x[:,0])
        model2 = auto_arima((4,1,2))
        model2.fit(x[:,1])
        model3 = auto_arima((4,1,2))
        model3.fit(x[:,2])
        
        datapoint.extend(model1.params().tolist())
        datapoint.extend(model2.params().tolist())
        datapoint.extend(model3.params().tolist())
        
        model = PCA(n_components = 3).fit(x)
        component1 = model.components_[0]
        component2 = model.components_[1]
        component3 = model.components_[2]
        pcas = (np.concatenate((component1,component2, component3)))
        datapoint.extend(pcas.tolist())
        
        datapoints.append(np.array(datapoint))
    return np.array(datapoints)

def transform_ts_to_datapoints2(X, bins):
    datapoints = []
    dim1 = X[0].shape[1]
    for x in X:
        datapoint = np.zeros((bins.shape))
        for xt in x:
            for i in range(dim1):
                for j,b in enumerate(bins[i]):
                    if xt[i] < b:
                        datapoint[i,j] += 1
                        break
        datapoints.append(np.array(datapoint).reshape(-1,1))
    return np.array(datapoints)[:,:,0]

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from gtda.time_series import TakensEmbedding
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

def fit_models(X, K, l, labels, splitidx):
    la = np.arange(K)
    seen = []
    b=0
    temp = np.copy(labels)
    for label in set(labels.tolist()):
        idx = np.where(temp == label)[0]
        labels[idx] = la[b]
        b+=1
    TE = TakensEmbedding(time_delay = l, dimension = 1)
    models = []
    for i in range(K):
        idx = np.where(labels==i)[0]
        curr_x = X[idx]
        predictors = np.array([])
        targets = np.array([])
        
        for x in curr_x:
            t = np.arange(x.shape[0])
            weights = np.exp(-(t-x.shape[0]//2)**2)
            x = x[:int(splitidx*x.shape[0])]
            predictor_part = x[:x.shape[0] - 1]*weights[:x.shape[0] - 1].reshape(-1,1)
            predictor_part
            target_part = x[1:]
            
            if targets.shape[0] == 0:
                targets = target_part
                predictors = predictor_part
                continue
            targets = np.concatenate((targets, target_part))
            predictors = np.concatenate((predictors, predictor_part))
        model = LinearRegression().fit(predictors, targets)
        models.append(model)
    return models
            
def gfc(X, K = 5, max_iter = 15, l =5, splitidx = .65, tol = .01, labels = None, eps = 0.5):
    X=np.array(X, dtype="object")
    if labels is None:
        labels = np.argmax([np.random.dirichlet(np.random.dirichlet(np.ones(K))) for _ in range(X.shape[0])], axis=1)
    models = fit_models(X, K, l, labels, splitidx)
    TE = TakensEmbedding(time_delay = l, dimension = 1)
    old_error = 100
    mean_errors = []
    for n in range(max_iter):
        all_errors = []
        for i,x in enumerate(X):
            t = np.arange(x.shape[0])
            weights = np.exp(-(t-x.shape[0]//2)**2)
            valset = (x[int(splitidx*x.shape[0]):])
            predictors = valset[:valset.shape[0] - 1]*weights[int(splitidx*x.shape[0]):x.shape[0] - 1].reshape(-1,1)
            targets = valset[1:]
            errors = []
            
            for j,model in enumerate(models):
                pred = model.predict(predictors)
                error = np.mean(np.abs(pred - targets))
                errors.append(error)
            
            A = np.random.choice([0,1],1,p=[eps, 1-eps])[0]
            if A:
                labels[i] = np.argmin(errors)
            else:
                labels[i] = np.random.choice(np.arange(K), 1)[0]
            all_errors.append(np.min(errors))
            
        old_error = np.mean(all_errors)
        mean_errors.append(old_error)
        K = len(set(labels))
        models = fit_models(X, K, l, labels, splitidx)
        eps=eps/(n+1)
    plt.plot(mean_errors)
    plt.show()
    return labels
from hmmlearn.hmm import GaussianHMM, GMMHMM
def fit_hmms(X, K, labels, dist = "GMM"):
    la = np.arange(K)
    seen = []
    b=0
    temp = np.copy(labels)
    for label in set(labels.tolist()):
        idx = np.where(temp == label)[0]
        labels[idx] = la[b]
        b+=1
    models = []
    for i in range(K):
        idx = np.where(labels==i)[0]
        curr_x = X[idx]
        seq = np.array([])
        lens = []
        for x in curr_x:
            lens.append(x.shape[0])
            if seq.shape[0] == 0:
                seq = x
                continue
            seq = np.concatenate((seq, x))
        if dist == "GMM":
            model = BayesianGaussianMixture(n_components = 3).fit(seq)
        elif dist == "HMM":
            model = GaussianHMM(n_components = 9).fit(seq, lengths = lens)
            
        models.append(model)
    return models

def HMM_clustering(X, K = 5, max_iter = 20, labels = None, splitidx = .8, eps = 0, dist = "GMM"):
    X=np.array(X, dtype="object")
    labels = []
    init_models = []
    for x in X:
        model = BayesianGaussianMixture(n_components = 3).fit(x)
        init_models.append(model)
        
    association_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i, model in enumerate(init_models):
        associated_series = np.zeros(X.shape[0])
        for j, x in enumerate(X):
            if j == i:
                associated_series[j] = 0
                continue
            associated_series[j] = np.exp(model.score(x))
        association_matrix[i,:] = associated_series
    
    labels = AgglomerativeClustering(n_clusters = K, distance_threshold =None,
                                         metric="precomputed", linkage="single").fit_predict(association_matrix)
    
    labels = np.argmax([np.random.dirichlet(np.random.dirichlet(np.ones(K))) for _ in range(X.shape[0])], axis=1)
    models = fit_hmms(X,K,labels, dist=dist)
    old_error = 100
    mean_errors=[]
    
    for m in range(max_iter):
        all_errors = []
        for i,x in enumerate(X):
            scores = []
            
            for j,model in enumerate(models):
                scores.append((model.score(x)))
                
            labels[i] = np.argmax(scores)
            all_errors.append(np.max(scores))
            
        old_error = np.mean(all_errors)
        mean_errors.append(old_error)
        K = len(set(labels))
        
        models = fit_hmms(X,K,labels, dist=dist)
        eps = eps/(m+1)
    plt.plot(mean_errors)
    plt.show()
    return labels

from sklearn.metrics import homogeneity_completeness_v_measure
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

def sigmoid(t):
    return 1/(1+np.exp(-t))

def bell_curve(t):
    return np.exp(-t**2)
        
OMP_NUM_THREADS = 1
X = []
X_no_window = np.array([])
labels = []
window_sizes = []

pi1 = np.array([1,0, 0])
pi2 = np.array([0, 1, 0])
pi3 = np.array([0,0,1])
pi4 = np.array([.6, .2, .2])
pi5 = np.array([.11, .09, .8])

means = np.array([[0,1,-1], [.5, .1, -.8], [0,0,0], [-1,-1, 3], [1,1,1]])*.001
pis = np.array([pi1, pi2, pi3, pi4, pi5])
cov = np.array([[1, .2, .2],
                [.2, 1, .2],
                [.2, .2, 1]])
cov=cov*.01

n_windows =  250
t=0

alpha = [[1,1,1,1], [.8,.9,1,.6], [1,.1,1,.3], [1,1,1,.1], [.2,.9,.7,.2]]
arma_scaling = .025
walk_scaling = .1

for _ in range(n_windows):
    S = np.random.choice([0,1,2,3,4], 1)[0]
    window_size = np.random.choice([50, 80, 100],1)[0]
    shift = np.random.choice(np.arange(30), 1)[0]
    t = np.arange(shift-window_size//2, shift+window_size//2)/60
    p = np.random.dirichlet(alpha[S], size = window_size)
    rw = random_walk(p, scaling=walk_scaling)
    
    if S == 0:
        f = f1(window_size)
        curr_window = np.random.multivariate_normal([0,0,0], cov, size=window_size)+ arma_scaling*f
        curr_window[:,0] += .8*bell_curve(t)+.1*sigmoid(t) + .1*np.log(1+np.abs(t))
        curr_window[:,1] += .9*bell_curve(t) + .15*np.abs(np.sin(t))
        curr_window[:,2] += sigmoid(t)
    elif S == 1:
        f = f2(window_size)
        curr_window = np.random.multivariate_normal([0,0,0], cov, size=window_size) + arma_scaling*f
        curr_window[:,0] += .3*sigmoid(t)+ .7*np.abs(np.sin(t))
        curr_window[:,1] += bell_curve(t)
        curr_window[:,2] += .5*bell_curve(t) + .5*sigmoid(t)
    elif S == 2:
        f = f3(window_size)
        curr_window = np.random.multivariate_normal([0,0,0], cov, size=window_size)+ arma_scaling*f
        curr_window[:,0] += bell_curve(t)
        curr_window[:,1] += .8*bell_curve(t) + .15*sigmoid(t)+ .1*np.abs(np.sin(t))
        curr_window[:,2] += bell_curve(t)
    elif S == 3:
        f = f4(window_size)
        curr_window = np.random.multivariate_normal([0,0,0], cov, size=window_size)+ arma_scaling*f
        curr_window[:,0] += .3*bell_curve(t)+ .7*np.abs(np.sin(t))
        curr_window[:,1] += bell_curve(t)+ np.arctan(t)*.1
        curr_window[:,2] += .5*np.arctan(t) + .5*sigmoid(t)
    elif S == 4:
        f = f5(window_size)
        curr_window = np.random.multivariate_normal([0,0,0], cov, size=window_size)+ arma_scaling*f
        curr_window[:,0] += .05*sigmoid(t)+ .9*np.abs(np.sin(t)) + bell_curve(t)*.05
        curr_window[:,1] += bell_curve(t) + sigmoid(t)*.01
        curr_window[:,2] += .2*bell_curve(t) + .5*sigmoid(t)+ np.arctan(t)*.3
    
    curr_window[:,0] += rw
    curr_window[:,1] += rw
    curr_window[:,2] += rw
    curr_window = curr_window/np.max(np.abs(curr_window), axis=0)
    #curr_window = rw.reshape(-1,1)
    curr_window = (curr_window)
    X.append(curr_window)
    window_sizes.append(window_size)
    labels.append(S)
    
plot_series(X, labels, 30, 0)
plot_series(X, labels, 10, 1)
plot_series(X, labels, 10, 2)
labels = np.array(labels)
label_order = []
for l in labels:
    if l in label_order:
        continue
    label_order.append(l)

 
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

'''
epochs = 5
outputs = []
losses = []
new_X = []
import umap.umap_ as umap
from sklearn.manifold import TSNE
for i,x in enumerate(X):
    if i%10==0:
        print(i)
        
    new_x = PCA(n_components=1).fit_transform(x)
    new_X.append(np.array(new_x))
new_X = X
plot_series(new_X, labels)
'''
new_X = X
from time import time

'''
nu = 1
dist_function = lambda x,y: (np.abs(np.dot(x-y,x-y)))**nu
distance_matrix = np.zeros((n_windows,n_windows))
st = time()
for i, ts1 in enumerate(X):
    print(i)
    for j, ts2 in enumerate(X):
        d, _, _, _ = dtw(ts1, ts2, dist = dist_function)
        distance_matrix[i,j] = d
print(time()-st)
'''
'''
models = []
A_init = np.array([[.8,.2],[.2,.8]])
means_init = np.array([[0,0,0],[-.1,-1,.5]])

for x in (X):
    model = GaussianHMM(n_components = 2, transmat_prior = A_init, means_prior=means_init, init_params = "sc", params = "stmc")
    model.transmat_ = A_init
    model.means_ = means_init
    print(x.shape)
    model.fit(x)
    models.append(model)
    
distance_matrix = np.zeros((n_windows,n_windows))
for i, ts1 in enumerate(models):
    print(i)
    for j, ts2 in enumerate(models):
        d = HMM_distance(ts1, ts2)
        distance_matrix[i,j] = d
'''

'''
inf = 99999
for i,x1 in enumerate(X):
    for j,x2 in enumerate(X):
        corr = np.corrcoef(x1, x2)
        d = 1/np.mean(corr)
        distance_matrix[i,j] = d
'''

time_table = {}
t = np.arange(-100, 100)/20
shapelet = np.random.multivariate_normal([0,0,0,0,0,0,0,0], np.eye(8)*.001, size=t.shape[0])
shapelet[:,0] += bell_curve(t)
shapelet[:,1] += sigmoid(t)
shapelet[:,2] += bell_curve(t) + sigmoid(t) + np.abs(np.sin(t))
shapelet[:,3] += .8*bell_curve(t)+.1*sigmoid(t) + .1*np.log(1+np.abs(t))
shapelet[:,4] += .9*bell_curve(t) + .15*np.abs(np.sin(t))
shapelet[:,5] += .3*sigmoid(t)+ .7*np.abs(np.sin(t))
shapelet[:,6] += bell_curve(t)+ np.arctan(t)*.1
shapelet[:,7] += bell_curve(t) + sigmoid(t)*.01
plt.plot(shapelet)
plt.show()

import stumpy
from sklearn.cluster import AgglomerativeClustering, DBSCAN
distance_matrix = np.zeros((n_windows,n_windows))
mp_distance = np.zeros((n_windows, n_windows))
st = time()
'''
for i,x1 in enumerate(new_X):
    for j,x2 in enumerate(new_X):
        if i == j:
            mp_distance[i,j] =0
            continue
        d = 0
        for dim in range(x1.shape[1]):
            d += stumpy.mpdist(x1[:,dim], x2[:,dim], 20)
        mp_distance[i,j] = d
mp_labels = AgglomerativeClustering(n_clusters =5, distance_threshold =None,
                                     metric="precomputed", linkage="complete").fit_predict(mp_distance)
'''
time_table["MP"] = time() - st

distance_matrix_euclid = np.zeros((n_windows,n_windows))
st = time()
'''
for i,x1 in enumerate(new_X):
    for j,x2 in enumerate(new_X):
        d = calc_euclid(x1, x2)
        distance_matrix_euclid[i,j] = d
'''
time_table["EUCLID"] = time() - st

st = time()
pred_labels = gfc(new_X, labels = None)
time_table["GFC"] = time() - st

st = time()
datapoints3 = ARIMA_error(X)
time_table["ARIMA"] = time() - st

st = time()
#distance_matrix3 = GMM_distance(new_X)
time_table["JSD"] = time() - st

labels_pred_euc = AgglomerativeClustering(n_clusters =5, distance_threshold =None,
                                     metric="precomputed", linkage="complete").fit_predict(distance_matrix_euclid)
#labels_pred4 = AgglomerativeClustering(n_clusters =5, distance_threshold =None,
#                                     metric="precomputed", linkage="complete").fit_predict(distance_matrix3)

st = time()
datapoints = transform_ts_to_datapoints(X)
time_table["DT1"] = time() - st

labels_pred3 = AgglomerativeClustering(n_clusters = 5, linkage="complete").fit_predict(datapoints)
labels_pred8 = AgglomerativeClustering(n_clusters = 5).fit_predict(datapoints3[:,:,0])

st = time()
labels_gmm = HMM_clustering(new_X)
time_table["GMM"] = time() - st

st = time()
labels_hmm = HMM_clustering(new_X, dist="HMM")
time_table["HMM"] = time() - st

st = time()
bin1 = np.arange(-1, 2, .1)
datapoints2 = transform_ts_to_datapoints2(X,np.array([bin1, bin1, bin1]))
time_table["DT2"] = time() - st
labels_pred5 = AgglomerativeClustering(n_clusters = 5).fit_predict(datapoints2)

labels_pred =AgglomerativeClustering(n_clusters =5, distance_threshold =None,
                                     metric="precomputed", linkage="complete").fit_predict(distance_matrix)
datapoints4 = []
st = time()
for x in X:
    model = PCA(n_components = 3).fit(x)
    component1 = model.components_[0]
    component2 = model.components_[1]
    component3 = model.components_[2]
    datapoints4.append(np.concatenate((component1,component2, component3)))
datapoints4 = np.array(datapoints4)
time_table["PCA"] = time() - st
labels_pred_pca = AgglomerativeClustering(n_clusters = 5, linkage="complete").fit_predict(datapoints4)  

st = time()
largest_window = 0
for x in X:
    if x.shape[0] > largest_window:
        largest_window = x.shape[0]

X_ae = np.zeros((n_windows, largest_window, 3))
for i,x in enumerate(X):
    padding = largest_window-x.shape[0]
    x = np.concatenate((x, np.zeros((padding, x.shape[1]))))
    X_ae[i] = x
X_ae = np.array(X_ae)
avg_losses = []
epochs = 5
model = AE()
optimizer = torch.optim.Adam(model.parameters(),
                                 lr = 1e-2,
                                 weight_decay = 1e-8)
for epoch in range(epochs):
    losses = []
    if epoch%10==0:
        print(epoch)
    for i,x in enumerate(X_ae):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.reshape(-1, 3*largest_window)
        reconstructed = model(x)
           
        loss = loss_function(reconstructed, x)
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_losses.append(np.mean(losses))
    
X_encoded = []
for xt in X_ae:
    xt = torch.tensor(xt, dtype=torch.float32)
    xt = xt.reshape(-1, 3*largest_window)
    x_encoded = model.encoder(xt).detach().numpy()
    X_encoded.append(x_encoded[0])
X_encoded = np.array(X_encoded)
pred_ae = KMeans(n_clusters=5).fit_predict(X_encoded)
time_table["AE"] = time() - st
v_scores = []
methods = ["AE", "ARIMA", "DT1", "DT2", "GFC", "GMM", "HMM", "EUCLID", "PCA", "TSKMEANS", "GAK"]
print("MP distance")
#v, _ = score(mp_labels, labels, label_order)
#v_scores.append(v[2])
print("AE + KMeans")
v,_ = score(pred_ae, labels, label_order)
v_scores.append(v[2])
print("ARIMA error")
v,_ = score(labels_pred8, labels, label_order)
v_scores.append(v[2])
print("GMM JSD")
#v,_ = score(labels_pred4, labels, label_order)
#v_scores.append(v[2])
print("Datapoints transformation")
v,_ = score(labels_pred3, labels, label_order)
v_scores.append(v[2])
print("Datapoints transformation 2")
v,_ = score(labels_pred5, labels, label_order)
v_scores.append(v[2])
print("GFC")
v,_ = score(pred_labels, labels, label_order)
v_scores.append(v[2])
print("GMM")
v,_ = score(labels_gmm, labels, label_order)
v_scores.append(v[2])
print("HMM")
v,_ = score(labels_hmm, labels, label_order)
v_scores.append(v[2])
print("Euclidean")
v,_ = score(labels_pred_euc, labels, label_order)
v_scores.append(v[2])
print("PCA")
v,_ = score(labels_pred_pca, labels, label_order)
v_scores.append(v[2])

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans
from tslearn.utils import to_time_series_dataset

st = time()
Xt = to_time_series_dataset(new_X)
tests = ['dtw']
pred = TimeSeriesKMeans(n_clusters=5, metric='dtw').fit_predict(Xt)
time_table["TSKMEANS"] = time() - st
print("TS KMeans")
v, pred = score(pred, labels, label_order)
v_scores.append(v[2])
#plot_series(X, pred)
st = time()
print("Kernelized KMeans with GAK")
v,_ = score(KernelKMeans(n_clusters=5).fit_predict(Xt), labels, label_order)
time_table["GAK"] = time() - st
v_scores.append(v[2])

print("TOP FIVE METHODS: ")
for i in range(5):
    curr_max = np.argmax(v_scores)
    print(f"{i+1}: Method {methods[curr_max]}, with V-score: {v_scores[curr_max]}, took: {time_table[methods[curr_max]]} seconds")
    v_scores[curr_max] = 0
