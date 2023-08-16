#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This file implements simulations routines 
for Ridge and Logistic regression models.
Authors: Rémi Leluc, François Portier
'''

import numpy as np
from models import linearReg,logisticReg
from tqdm.notebook import tqdm

def run_ols(X,y,𝜆,method,N_exp,N,burn_in,batch_size,batch_hess,
            lr,c,k0,a,alpha_power,𝜂,power,same):
    n_features = X.shape[1]
    w_res = np.zeros((N_exp,N+1,n_features))
    w0 = np.zeros(n_features)
    np.random.seed(0)
    loss_list = np.zeros((N_exp,N+1))
    model_ols = linearReg(X=X,y=y,𝜆=𝜆)
    for i in tqdm(range(N_exp)):
        w_res_list = model_ols.fit(seed=i,method=method,N=N,w0=w0,burn_in=burn_in,
                                   batch_size=batch_size,
                                   batch_hess=batch_hess,lr=lr,c=c,
                                   k0=k0,a=a,alpha_power=alpha_power,𝜂=𝜂, power=power,same=same)
        w_res[i] = w_res_list
        loss = [model_ols.loss(w) for w in w_res_list]
        loss_list[i] = loss
    return w_res,loss_list


def run_log(X,y,𝜆,method,N_exp,N,burn_in,batch_size,batch_hess,
            lr,c,k0,a,alpha_power,𝜂,power,fit_intercept,same):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    if fit_intercept:
        w0 = np.zeros(n_features+1)
        w_res = np.zeros((N_exp,N+1,n_features+1))
    else:
        w0 = np.zeros(n_features)
        w_res = np.zeros((N_exp,N+1,n_features))
    loss_list = np.zeros((N_exp,N+1))
    model_log = logisticReg(X=X,y=y,𝜆=𝜆,fit_intercept=fit_intercept)
    for i in tqdm(range(N_exp)):
        w_res_list = model_log.fit(seed=i,method=method,N=N,w0=w0,burn_in=burn_in,batch_size=batch_size,
                                   batch_hess=batch_hess,lr=lr,c=c,
                                   k0=k0,a=a,alpha_power=alpha_power,𝜂=𝜂, power=power,same=same)
        loss = [model_log.loss(w) for w in w_res_list]
        w_res[i] = w_res_list
        loss_list[i] = loss
    return w_res,loss_list
