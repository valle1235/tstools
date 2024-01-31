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

class AE(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4*input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16)
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4*input_size),
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

def GMM_distance(X):
    n_windows= len(X)
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
        model1 = ARIMA((4,0,0))
        model1.fit(x[:,0])
        model2 = ARIMA((4,0,0))
        model2.fit(x[:,1])
        model3 = ARIMA((4,0,0))
        model3.fit(x[:,2])
        
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
        
        model1 = ARIMA((4,0,0))
        model1.fit(x[:,0])
        model2 = ARIMA((4,0,0))
        model2.fit(x[:,1])
        model3 = ARIMA((4,0,0))
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
#from gtda.time_series import TakensEmbedding
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from pygam import LinearGAM, s

def fit_models(X, K, l, labels, splitidx):
    la = np.arange(K)
    seen = []
    b=0
    temp = np.copy(labels)
    for label in set(labels.tolist()):
        idx = np.where(temp == label)[0]
        labels[idx] = la[b]
        b+=1
    #TE = TakensEmbedding(time_delay = l, dimension = 1)
    models = []
    for i in range(K):
        idx = np.where(labels==i)[0]
        curr_x = X[idx]
        predictors = np.array([])
        targets = np.array([])
        
        for x in curr_x:
            t = np.arange(x.shape[0])
            #weights = np.exp(-(t-x.shape[0]//2)**2)
            x = x[:int(splitidx*x.shape[0])]
            predictor_part = x[:x.shape[0] - 1]
            predictor_part
            target_part = x[1:,0]
            
            if targets.shape[0] == 0:
                targets = target_part
                predictors = predictor_part
                continue
            targets = np.concatenate((targets, target_part))
            predictors = np.concatenate((predictors, predictor_part))
        model = LinearRegression().fit(predictors, targets)
        models.append(model)
    return models
            
def gfc(X, K = 5, max_iter = 100, l =5, splitidx = .65, tol = .01, labels = None, eps = 0.5):
    X=np.array(X, dtype="object")
    if labels is None:
        labels = np.argmax([np.random.dirichlet(np.random.dirichlet(np.ones(K))) for _ in range(X.shape[0])], axis=1)
    models = fit_models(X, K, l, labels, splitidx)
    #TE = TakensEmbedding(time_delay = l, dimension = 1)
    old_error = 100
    mean_errors = []
    for n in range(max_iter):
        all_errors = []
        for i,x in enumerate(X):
            t = np.arange(x.shape[0])
            #weights = np.exp(-(t-x.shape[0]//2)**2)
            valset = (x[int(splitidx*x.shape[0]):])
            predictors = valset[:valset.shape[0] - 1]
            targets = valset[1:,0]
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
from scipy.stats import entropy

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
            model = BayesianGaussianMixture(n_components = 6).fit(seq)
        elif dist == "HMM":
            model = GaussianHMM(n_components = 9).fit(seq, lengths = lens)
            
        models.append(model)
    return models

def HMM_clustering(X, K = 5, max_iter = 30, labels = None, splitidx = .8, eps = 0.15, dist = "GMM"):
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
    
    #labels = np.argmax([np.random.dirichlet(np.random.dirichlet(np.ones(K))) for _ in range(X.shape[0])], axis=1)
    models = fit_hmms(X,K,labels, dist=dist)
    old_error = 100
    mean_errors=[]
    
    for m in range(max_iter):
        all_errors = []
        for i,x in enumerate(X):
            scores = []
            
            for j,model in enumerate(models):
                scores.append((model.score(x)))
                
            #labels[i] = np.argmax(scores)
            #all_errors.append(np.max(scores))
            A = np.random.choice([0,1],1,p=[eps, 1-eps])[0]
            if A:
                labels[i] = np.argmax(scores)
            else:
                labels[i] = np.random.choice(np.arange(K), 1)[0]
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
from generate_ts import generate, generate_impure_signals,generate_pdw, generate_batches
from deinterleave import deinterleave
from matrixprofile.algorithms import mpdist
def main():
    n_windows = 100
    X, labels = generate_batches(n_windows, 800, p = [.35, .2, .15, .15, .15])
    X1, X2, sizes, X, labels = deinterleave(X,labels)
    n_windows = 5*n_windows
    print(len(X))
    #for i in range(len(X)):
    #    X[i] = quantile_transform(X[i], n_quantiles=50)
    
    '''
    X_complete = np.array([])
    for x in X:
        if X_complete.shape[0] == 0:
            X_complete = x
            continue
        X_complete = np.append(X_complete, x, axis = 0)
    X_complete = quantile_transform(X_complete, n_quantiles = 750)
    transformed_X = []
    t = 0
    for x in X:
        transformed_X.append(X_complete[t:t+x.shape[0]])
        t+=x.shape[0]
    X = transformed_X
    '''
    for dim in range(X[0].shape[1]):
        plot_series(X, labels, 5, dim)
    labels = np.array(labels)
    label_order = []
    for l in labels:
        if l in label_order:
            continue
        label_order.append(l)
    
     
    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()
    
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
    plot_series(new_X,labels, 10, 0)
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
    
    distance_matrix = np.zeros((n_windows,n_windows))
    mp_distance = np.zeros((n_windows, n_windows))
    st = time()
    for i,x1 in enumerate(new_X):
        for j,x2 in enumerate(new_X):
            if i == j:
                mp_distance[i,j] =0
                continue
            d = 0
            for dim in range(x1.shape[1]):
                d += mpdist(x1[:,dim], x2[:,dim], 5, n_jobs = -1)
            mp_distance[i,j] = d
    mp_labels = AgglomerativeClustering(n_clusters =5, distance_threshold =None,
                                         metric="precomputed", linkage="complete").fit_predict(mp_distance)
    
    time_table["MP"] = time() - st
    
    distance_matrix_euclid = np.zeros((n_windows,n_windows))
    st = time()
    
    for i,x1 in enumerate(new_X):
        for j,x2 in enumerate(new_X):
            d = corr_dist(x1, x2)
            distance_matrix_euclid[i,j] = d
    
    time_table["EUCLID"] = time() - st
    
    st = time()
    pred_labels = gfc(new_X, labels = None)
    time_table["GFC"] = time() - st
    
    st = time()
    datapoints3 = ARIMA_error(X)
    time_table["ARIMA"] = time() - st
    
    st = time()
    #distance_matrix3 = GMM_distance(new_X)
    distance_matrix3 = distance_matrix_euclid
    time_table["JSD"] = time() - st
    
    labels_pred_euc = AgglomerativeClustering(n_clusters =5, distance_threshold =None,
                                         metric="precomputed", linkage="complete").fit_predict(distance_matrix_euclid)
    labels_pred4 = AgglomerativeClustering(n_clusters =5, distance_threshold =None,
                                         metric="precomputed", linkage="complete").fit_predict(distance_matrix3)
    
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
    
    X_ae = np.zeros((n_windows, largest_window, 4))
    for i,x in enumerate(X):
        padding = largest_window-x.shape[0]
        x = np.concatenate((x, np.zeros((padding, x.shape[1]))))
        X_ae[i] = x
    X_ae = np.array(X_ae)
    avg_losses = []
    epochs = 5
    model = AE(largest_window)
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr = 1e-2,
                                     weight_decay = 1e-8)
    for epoch in range(epochs):
        losses = []
        if epoch%10==0:
            print(epoch)
        for i,x in enumerate(X_ae):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.reshape(-1, 4*largest_window)
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
        xt = xt.reshape(-1, 4*largest_window)
        x_encoded = model.encoder(xt).detach().numpy()
        X_encoded.append(x_encoded[0])
    X_encoded = np.array(X_encoded)
    pred_ae = KMeans(n_clusters=5).fit_predict(X_encoded)
    time_table["AE"] = time() - st
    v_scores = []
    methods = ["MP", "AE", "ARIMA","JSD", "DT1", "GFC", "GMM", "HMM", "EUCLID", "PCA", "TSKMEANS", "GAK"]
    print("MP distance")
    v, _ = score(mp_labels, labels, label_order)
    v_scores.append(v[2])
    print("AE + KMeans")
    v,_ = score(pred_ae, labels, label_order)
    v_scores.append(v[2])
    print("ARIMA error")
    v,_ = score(labels_pred8, labels, label_order)
    v_scores.append(v[2])
    print("GMM JSD")
    v,_ = score(labels_pred4, labels, label_order)
    v_scores.append(v[2])
    print("Datapoints transformation")
    v,_ = score(labels_pred3, labels, label_order)
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

if __name__ == "__main__":
    main()
