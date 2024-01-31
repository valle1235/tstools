# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 05:40:59 2024

@author: Valdemar
"""

import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

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

def sigmoid(t):
    return 1/(1+np.exp(-t))

def bell_curve(t):
    return np.exp(-t**2)

def generate(n_windows, walk_scaling = .1, arma_scaling = .1):
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
        
    alpha = [[1,1,1,1], [.8,.9,1,.6], [1,.1,1,.3], [1,1,1,.1], [.2,.9,.7,.2]]
    
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
        
    return X, labels

def generate_impure_signals(n_windows, walk_scaling = .1, arma_scaling = .1, impurity = .2, ipl = 10, ws = [300, 500, 700]):
    X = []
    X_no_window = np.array([])
    labels = []
    window_sizes = []
    p = impurity/4
    
    pi1 = np.array([1-impurity,p,p,p,p])
    pi2 = np.array([p,1-impurity,p,p,p])
    pi3 = np.array([p,p,1-impurity,p,p])
    pi4 = np.array([p,p,p,1-impurity,p])
    pi5 = np.array([p,p,p,p,1-impurity])
    
    means = np.array([[0,1,-1], [.5, .1, -.8], [0,0,0], [-1,-1, 3], [1,1,1]])*.001
    pis = np.array([pi1, pi2, pi3, pi4, pi5])
    cov = np.array([[1, .2, .2],
                    [.2, 1, .2],
                    [.2, .2, 1]])
    cov=cov*.01
        
    alpha = [[1,1,1,1], [.8,.9,1,.6], [1,.1,1,.3], [1,1,1,.1], [.2,.9,.7,.2]]
    
    for _ in range(n_windows):
        state = np.random.choice([0,1,2,3,4], 1)[0]
        window_size = np.random.choice(ws,1)[0]
        shift = np.random.choice(np.arange(30), 1)[0]
        all_t = np.arange(shift-window_size//2, shift+window_size//2)/60
        pi = pis[state]
        
        curr_window = np.zeros((window_size,3))
        for offset in range(0,all_t.shape[0],ipl):
            t = all_t[offset:offset+ipl]
            S = np.random.choice([0,1,2,3,4], 1, p=pi)[0]
            p = np.random.dirichlet(alpha[S], size = ipl)
            rw = random_walk(p, scaling=walk_scaling)
            
            if S == 0:
                f = f1(ipl)
                temp = np.random.multivariate_normal([0,0,0], cov, size=ipl)+ arma_scaling*f
                temp[:,0] += .8*bell_curve(t)+.1*sigmoid(t) + .1*np.log(1+np.abs(t))
                temp[:,1] += .9*bell_curve(t) + .15*np.abs(np.sin(t))
                temp[:,2] += sigmoid(t)
            elif S == 1:
                f = f2(ipl)
                temp = np.random.multivariate_normal([0,0,0], cov, size=ipl) + arma_scaling*f
                temp[:,0] += .3*sigmoid(t)+ .7*np.abs(np.sin(t))
                temp[:,1] += bell_curve(t)
                temp[:,2] += .5*bell_curve(t) + .5*sigmoid(t)
            elif S == 2:
                f = f3(ipl)
                temp = np.random.multivariate_normal([0,0,0], cov, size=ipl)+ arma_scaling*f
                temp[:,0] += bell_curve(t)
                temp[:,1] += .8*bell_curve(t) + .15*sigmoid(t)+ .1*np.abs(np.sin(t))
                temp[:,2] += bell_curve(t)
            elif S == 3:
                f = f4(ipl)
                temp = np.random.multivariate_normal([0,0,0], cov, size=ipl)+ arma_scaling*f
                temp[:,0] += .3*bell_curve(t)+ .7*np.abs(np.sin(t))
                temp[:,1] += bell_curve(t)+ np.arctan(t)*.1
                temp[:,2] += .5*np.arctan(t) + .5*sigmoid(t)
            elif S == 4:
                f = f5(ipl)
                temp = np.random.multivariate_normal([0,0,0], cov, size=ipl)+ arma_scaling*f
                temp[:,0] += .05*sigmoid(t)+ .9*np.abs(np.sin(t)) + bell_curve(t)*.05
                temp[:,1] += bell_curve(t) + sigmoid(t)*.01
                temp[:,2] += .2*bell_curve(t) + .5*sigmoid(t)+ np.arctan(t)*.3
            
            temp[:,0] += rw
            temp[:,1] += rw
            temp[:,2] += rw
            curr_window[offset:offset+ipl] = temp
            
        curr_window = curr_window/np.max(np.abs(curr_window), axis=0)
        X.append(curr_window)
        window_sizes.append(window_size)
        labels.append(state)
        
    return X, labels

def generate_pdw(n_windows, impurity = .1, ws = [300, 500, 700], ipl = 10):
    X = []
    labels = []
    t = 0
    
    p = impurity/4
    
    pi1 = np.array([1-impurity,p,p,p,p])
    pi2 = np.array([p,1-impurity,p,p,p])
    pi3 = np.array([p,p,1-impurity,p,p])
    pi4 = np.array([p,p,p,1-impurity,p])
    pi5 = np.array([p,p,p,p,1-impurity])
    
    rf_means = np.array([0,.25,.5,.75,1])
    amp_scalings = np.array([.4,.5,.6,.7,.8])
    amp_freq = np.array([1/100, 1/200, 1/150, 1/80, 1/50])
    aoa_start = np.array([100,98,94,90,86])
    aoa_coef = np.array([10**(-7),10**(-8),10**(-7.5),10**(-9),10**(-8.5)])
    pw_start = np.array([1.1, 1.2, 1.9, 2, 0.2])
    alpha = [[1,1,1,1], [.8,.9,1,.6], [1,.1,1,.3], [1,1,1,.1], [.2,.9,.7,.2]]
    
    pis = np.array([pi1, pi2, pi3, pi4, pi5])
    cov = np.array([[1, .2, .2, .2],
                    [.2, 1, .2, .2],
                    [.2, .2, 1, .2],
                    [.2, .2, .2, 1]])
    cov=cov*.001
    
    for _ in range(n_windows):
        state = np.random.choice([0,1,2,3,4], 1)[0]
        window_size = np.random.choice(ws,1)[0]
        pi = pis[state]
        
        curr_window = np.zeros((window_size,4))
        time_stamps = np.arange(t, t+window_size)
        
        #[rf, PW, AoA, Amp]
        for offset in range(0,time_stamps.shape[0],ipl):
            curr_t = time_stamps[offset:offset+ipl]
            S = np.random.choice([0,1,2,3,4], 1, p=pi)[0]
            p = np.random.dirichlet(alpha[S], size = ipl)
            
            rf = rf_means[state]
            amp = amp_scalings[state]*np.sinc(((curr_t % 130) - 100)*amp_freq[state])
            aoa = aoa_start[state] - aoa_coef[state]*curr_t**2
            pw = random_walk(p, scaling=pw_start[state])
            
            curr_window[offset:offset+ipl,0] = rf + (S-state)*.05
            curr_window[offset:offset+ipl,1] = amp + (S-state)*.05
            curr_window[offset:offset+ipl,2] = aoa%360 -180 + (S-state)*.05
            curr_window[offset:offset+ipl,3] = pw + (S-state)*.05
        
        t+=window_size
        X.append(curr_window + np.random.multivariate_normal([0,0,0,0], cov, size=window_size))
        labels.append(state)
        
    return X, labels

def generate_batches(n_windows, batch_size, p = [.4, .1, .25, .15, .1]):
    X = []
    labels = []
    
    rf_means = np.array([0,.25,.5,.75,1])
    amp_scalings = np.array([.4,.5,.6,.7,.8])
    amp_freq = np.array([1/10, 1/30, 1/50, 1/40, 1/20])
    aoa_start = np.array([100,99,98,97,96])
    aoa_coef = np.array([10**(-5),10**(-4),10**(-6),10**(-5.5),10**(-4.5)])
    pw_start = np.array([1.1, 1.2, 1.9, 2, 0.2])
    alpha = [[1,1,1,1], [.8,.9,1,.6], [1,.1,1,.3], [1,1,1,.1], [.2,.9,.7,.2]]
    
    cov = np.array([[1, .2, .2, .2],
                    [.2, 1, .2, .2],
                    [.2, .2, 40, .2],
                    [.2, .2, .2, 1]])
    cov=cov*.001
    t = [0,0,0,0,0]
    for _ in range(n_windows):
        curr_window = np.zeros((batch_size,4))
        label_window = []
        for b in range(batch_size):
            S = np.random.choice(np.arange(5),1, p=p)[0]
            
            rf = rf_means[S]
            amp = amp_scalings[S]*np.sinc(((t[S] % 130) - 100)*amp_freq[S])
            aoa = aoa_start[S] - aoa_coef[S]*t[S]**2
            pw = np.random.normal(pw_start[S], .2)
            
            curr_window[b,0] = rf
            curr_window[b,1] = amp
            curr_window[b,2] = aoa%360 -180
            curr_window[b,3] = pw
            label_window.append(S)
            t[S] += 1
            reset_t = np.random.choice([0,1], 1, p = [.99, .01])[0]
            if reset_t:
                t[S] = 0
        
        labels.append(label_window)
        X.append(curr_window+ np.random.multivariate_normal([0,0,0,0], cov, size=batch_size))
        
    return X, labels