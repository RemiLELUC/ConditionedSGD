#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This file implements Ridge and Logistic regression models.
Authors: Rémi Leluc, François Portier
'''

import numpy as np
from fit_methods import optimize

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import warnings
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

class linearReg:
    def __init__(self, X, y, 𝜆):
        self.X = X  # data matrix
        self.y = y  # regression labels
        self.𝜆 = 𝜆
        self.n = X.shape[0] # n_samples
        self.d = X.shape[1] # n_features

    def loss(self,w):
        ''' Loss function OLS: L(w) = (1/2n) || Y - Xw ||^2  + (𝜆/2) ||w||^2'''
        data_term = np.sum((self.y-self.X.dot(w))**2)/(2*self.n)
        reg = (self.𝜆/2) * sum(w**2)
        return data_term + reg

    def batch_grad(self,batch,w):
        ''' Batch SG: g(w) = (1/|B|) X_b^T (X_bw-Y_b) '''
        X_b = self.X[batch]
        err = X_b.dot(w)-self.y[batch]
        grad = (X_b.T).dot(err)
        return grad/len(batch) + self.𝜆*w
    
    def batch_hessian(self,batch,w):
        ''' Batch Hessian: g(w)_k = (1/|B|) {X_b^T (X_b} + 𝜆 I_d '''
        Ip = np.eye(self.X.shape[1])
        B = len(batch)
        if B==0:
            H_batch = Ip
        else:
            X_b = self.X[batch]
            H_batch = (1/B)*(X_b.T).dot(X_b) + self.𝜆*np.eye(self.d)
        return H_batch

    def fit(self, seed, method, N, w0, burn_in,
            batch_size,batch_hess, lr,c,k0,a,alpha_power, 𝜂, power,same):
        ''' Fit Logistic regression model with Stochastic Gradient methods
             w_{k+1} = w_{k} - 𝛼_{k+1} C_k gradient(w_{k})
        where C_k = (\Phi_k + 𝛾_{k+1}^{-1}Ip)^{-1},    𝛾_k = k^power
        -(sgd)   \Phi_k = Ip
        -(C-sgd) \Phi_k = \sum_{j=0}^{k} \nu_{j,k} H_j
        '''
        return optimize(seed=seed,method=method,n=self.X.shape[0],
                        d=self.X.shape[1],N=N,w0=w0,burn_in=burn_in,
                        grad=self.batch_grad,
                        hess=self.batch_hessian,
                        batch_size=batch_size,batch_hess=batch_hess,lr=lr,c=c,k0=k0,a=a,
                        alpha_power=alpha_power, 𝜂=𝜂, power=power,same=same)

class logisticReg:
    def __init__(self, X, y, 𝜆, fit_intercept=False):
        self.X = X               # data matrix
        self.y = y               # binary label 0-1
        self.𝜆 = 𝜆               # regularization parameter
        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            self.add_intercept()
    
    def add_intercept(self):
        ''' Add column of 1 for intercept '''
        intercept = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((intercept, self.X), axis=1)
    
    def sigmoid(self, z):
        ''' Sigmoid function '''
        return 1 / (1 + np.exp(-z))
    
    def loss(self,w):
        ''' Regularized objective function at point w 
        for i=1,...,n, proba \pi_i = sigmoid(X_i^T w)
        loss(w) = -(1/n) \sum_{i=1}^n [y_i log(\pi_i) + (1-y_i) log(1-\pi_i)] + (𝜆/2)|w|^2
        Params:
        @w (array px1): point to evaluate
        Returns:
        @loss  (float): penalized loss function at w
        '''
        z = np.dot(self.X,w)
        # Negative Log Likelihood
        data_term = np.log(1+np.exp(np.multiply(-self.y,z))).mean()
        # Regulatization term (Ridge penalization)
        reg = (self.𝜆/2) * sum(w**2)
        return data_term + reg
    
    def batch_grad(self,batch,w):
        ''' Batch Gradient of regularized objective function
        Params:
        @batch (array Bx1): indices of the batch
        @w     (array px1): point to evaluate
        Returns:
        @gradient_batch (array px1): batch gradient at w
        '''
        B = len(batch)
        X_b = self.X[batch]
        z_b = np.dot(X_b, w)
        h_b = self.sigmoid(z_b)
        g = (1/B)*np.dot(X_b.T, (h_b - self.y[batch]))
        #g0 = g[0]
        #g1 = g[1:] + self.𝜆*w[1:]
        #gradient_batch = np.vstack((g0.reshape(-1,1),g1.reshape(-1,1))).ravel()
        gradient_batch = g + self.𝜆*w
        return gradient_batch
    
    def full_grad(self,w):
        ''' Full Gradient of regularized objective function
        Params:
        @w        (array px1): point to evaluate
        Returns:
        @gradient (array px1): full gradient at w
        '''
        n = self.y.size
        full = np.arange(n)
        return self.batch_grad(batch=full,w=w)
    
    def batch_hessian(self,batch,w):
        ''' Batch Hessian of regularized objective function
        Params:
        @batch   (array Bx1): indices of the batch
        @w       (array px1): point to evaluate
        Returns:
        @H_batch (array pxp): batch hessian at w
        '''
        Ip = np.eye(self.X.shape[1])
        B = len(batch)
        if B==0:
            H_batch = Ip
        else:
            X_b = self.X[batch]
            z_b = np.dot(X_b, w)
            h_b = self.sigmoid(z_b)
            H_batch = (1/B)*np.dot(np.dot(X_b.T,np.diag(h_b*(1-h_b))),X_b) + self.𝜆*Ip
            #H_batch[0,0] -= self.𝜆
        return H_batch
    
    def full_hessian(self,w):
        ''' Full Hessian of regularized objective function
        Params:
        @batch  (array Bx1): indices of the batch
        @w      (array px1): point to evaluate
        Returns:
        @H_full (array pxp): full hessian at w
        '''
        n = self.y.size
        full = np.arange(n)
        return self.batch_hessian(batch=full,w=w)
    
    
    
    def fit(self, seed, method, N, w0,burn_in,
            batch_size,batch_hess, lr,c,k0,a, alpha_power, 𝜂, power,same):
        ''' Fit Logistic regression model with Stochastic Gradient methods
                 w_{k+1} = w_{k} - 𝛼_{k+1} C_k gradient(w_{k})
        where C_k = (\Phi_k + 𝛾_{k+1}^{-1}Ip)^{-1},    𝛾_k = k^power
        -(sgd)   \Phi_k = Ip
        -(C-sgd) \Phi_k = \sum_{j=0}^{k} \nu_{j,k} H_j
        '''
        return optimize(seed=seed,method=method,n=self.X.shape[0],
                        d=self.X.shape[1],N=N,w0=w0,burn_in=burn_in,grad=self.batch_grad,
                        hess=self.batch_hessian,
                        batch_size=batch_size,batch_hess=batch_hess,
                        lr=lr,c=c,k0=k0,a=a,
                        alpha_power=alpha_power, 𝜂=𝜂, power=power,same=same)

    
    def predict_prob(self,X_test):
        ''' Predict probabilities given X_test: y_pred = sigmoid(X^T w_final)
        Params:
        @X_test (array n_test x p): data to predict
        Returns
        @y_pred (array n_test x 1): probabilities array
        '''
        if self.fit_intercept:
            intercept = np.ones((X_test.shape[0], 1))
            X_pred = np.concatenate((intercept, X_test), axis=1)
        else:
            X_pred = X_test
        return self.sigmoid(np.dot(X_pred, self.w))
    
    def predict(self,X_test,threshold=0.5):
        ''' Predict binary labels given X_test
        Params:
        @X_test (array n_test x p): data to predict
        @threshold (float in O,1): threshold for classification, default 0.5
        Returns
        @y_pred (array n_test x 1): binary array
        '''
        return (self.predict_prob(X_test=X_test) >= threshold).astype(int) 
