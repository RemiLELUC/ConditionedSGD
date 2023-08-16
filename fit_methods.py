#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This file implements (C)-SGD methods.
Authors: Rémi Leluc, François Portier
'''

import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv, pinv

def optimize(seed, method, n, d, N, w0, burn_in,
             grad, hess, batch_size,batch_hess, lr,c,
             k0,a, alpha_power, 𝜂, power,same):
    ''' Averaging at previous points with/without weights
    Params:
    @seed          (int): random seed for reproducibility
    @method     (string): 'sgd','sgd-avg','csgd','csgd-avg'
    @n             (int): number of samples
    @d             (int): dimension of the problem
    @N             (int): number of iterations
    @w0          (array): initial point
    @burn_in       (int): burn_in phase for averaging
    @grad         (func): gradient generator
    @hess         (func): hessian generator
    @batch_size    (int): for gradient estimates
    @batch_hess    (int): for hessian estimates
    @lr          (float): numerator learning rate
    @k0          (float): smoothing parameter denominator learning rate
    @a           (float): aI added to G0 in AdaFull_avg
    @alpha_power (float): power learning rate
    @c           (float): numerator learning rate C_k
    @𝜂           (float): coefficient in adaptive weights exp(-𝜂|w_j-w_k|^2)
    @power       (float): power of C_k learning rate
    @same         (bool): whether to take same batch for gradient and hessian estimates
    '''
    np.random.seed(seed) # set random seed
    w = w0.copy()
    w_bar = w0.copy()
    w_list = [w.copy()] # iterates evolution
    Ip = np.eye(d)
    G = a*np.eye(d)
    H_list = []
    if method=='sgd':
        batch_list = np.random.choice(a=np.arange(n),size=(N,batch_size))  
        for k in range(1,N+1):
            # Pick random batch
            batch = batch_list[k-1]
            # Compute Batch  Gradient
            gradient_batch = grad(batch=batch,w=w)
            # optimal step size 1/k 
            step = lr *(1/(k+k0))**alpha_power
            # update parameter batch gradient
            w -= step * gradient_batch
            # Save current iterate
            w_list.append(w.copy())
    if method=='sgd-avg':
        batch_list = np.random.choice(a=np.arange(n),size=(N,batch_size))  
        for k in range(1,N+1):
            # Pick random batch
            batch = batch_list[k-1]
            # Compute Batch  Gradient
            gradient_batch = grad(batch=batch,w=w)
            # optimal step size 1/k 
            step = lr *(1/(k+k0))**alpha_power
            # update parameter batch gradient
            w -= step * gradient_batch
            # average iterate
            if k<burn_in:
                w_bar = w
            else:
                w_bar = w_bar + (w-w_bar)/(k-burn_in+1)
            # Save current iterate
            w_list.append(w_bar.copy())
    if method=='adafull':
        batch_list = np.random.choice(a=np.arange(n),size=(N,batch_size))  
        for k in range(1,N+1):
            # Pick random batch
            batch = batch_list[k-1]
            # Compute Batch  Gradient
            gradient_batch = grad(batch=batch,w=w).reshape(-1,1)
            G += np.dot(gradient_batch,gradient_batch.T)
            evalues, evectors = np.linalg.eigh(G)
            C = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)
            d = np.linalg.solve(C,gradient_batch).ravel() 
            # update parameter 
            step = lr *(1/(k+k0))**alpha_power
            w -= step * d
            w_list.append(w.copy())
    if method=='adafull-avg':
        batch_list = np.random.choice(a=np.arange(n),size=(N,batch_size))  
        for k in range(1,N+1):
            # Pick random batch
            batch = batch_list[k-1]
            # Compute Batch  Gradient
            gradient_batch = grad(batch=batch,w=w).reshape(-1,1)
            G = G + (np.dot(gradient_batch,gradient_batch.T)-G)/(k+1)
            # optimal step size 1/k 
            evalues, evectors = np.linalg.eigh(G)
            C = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)
            d = np.linalg.solve(C,gradient_batch).ravel() 
            # update parameter 
            step = lr *(1/(k+k0))**alpha_power
            w -= step * d
            w_list.append(w.copy())
    if method=='adadiag':
        batch_list = np.random.choice(a=np.arange(n),size=(N,batch_size))  
        for k in range(1,N+1):
            # Pick random batch
            batch = batch_list[k-1]
            # Compute Batch  Gradient
            gradient_batch = grad(batch=batch,w=w).reshape(-1,1)
            M = np.dot(gradient_batch,gradient_batch.T)
            G = G + np.diag(np.diag(M))
            C = np.sqrt(G)
            # optimal step size 1/k 
            d = np.linalg.solve(C,gradient_batch.ravel()) 
            # update parameter 
            step = lr *(1/(k+k0))**alpha_power
            w -= step * d
            w_list.append(w.copy())
    if method=='adadiag-avg':
        batch_list = np.random.choice(a=np.arange(n),size=(N,batch_size))  
        for k in range(1,N+1):
            # Pick random batch
            batch = batch_list[k-1]
            # Compute Batch  Gradient
            gradient_batch = grad(batch=batch,w=w).reshape(-1,1)
            M = np.dot(gradient_batch,gradient_batch.T)
            G = G + (np.diag(np.diag(M))-G)/(k+1)
            C = np.sqrt(G)
            # optimal step size 1/k 
            d = np.linalg.solve(C,gradient_batch.ravel()) 
            # update parameter 
            step = lr *(1/(k+k0))**alpha_power
            w -= step * d
            w_list.append(w.copy())
    elif method=='csgd':
        # Initial batch of size 0 for Identity matrix
        batch_list_grad = np.random.choice(a=np.arange(n),size=(N,batch_size))  
        batch_prev = np.random.choice(a=n,size=0,replace=False)
        if same:
            batch_list = batch_list_grad
        else:
            batch_list = np.random.choice(a=np.arange(n),size=(N,batch_hess))  
        for k in range(1,N+1):
            # Pick random batch
            batch_grad = batch_list_grad[k-1]
            batch = batch_list[k-1]
            # Compute Batch Gradient
            gradient_batch = grad(batch=batch_grad,w=w)
            # Compute Batch Hessian with Batch from previous step
            H_b = hess(batch=batch_prev,w=w)
            #H_b = self.batch_fisher(batch=b,w=self.w.copy())
            # Store Hessian for later averaging
            H_list.append(H_b)
            # Convert to array for weighted averaging
            H_to_avg = np.array(H_list)
            # Compute weights for the average
            w_curr = w_list[-1]
            weight = np.exp(-𝜂*np.linalg.norm(x=np.array(w_list)-w_curr,ord=2,axis=1))
            # Compute weighted average of batch Hessian 
            H = np.average(H_to_avg,axis=0,weights=weight)
            # Conditioning matrix C
            C = H + np.power(c/k,power) * Ip
            # descent direction
            d = np.linalg.solve(C,gradient_batch) 
            # update parameter 
            step = lr *(1/(k+k0))**alpha_power
            w -= step * d
            # save current batch for next hessian estimate
            batch_prev = batch
            # Save current iterate
            w_list.append(w.copy())
    elif method=='csgd-avg':
        # Initial batch of size 0 for Identity matrix
        batch_list_grad = np.random.choice(a=np.arange(n),size=(N,batch_size))  
        batch_prev = np.random.choice(a=n,size=0,replace=False)
        if same:
            batch_list = batch_list_grad
        else:
            batch_list = np.random.choice(a=np.arange(n),size=(N,batch_hess))  
        for k in range(1,N+1):
            # Pick random batch
            batch_grad = batch_list_grad[k-1]
            batch = batch_list[k-1]
            # Compute Batch Gradient
            gradient_batch = grad(batch=batch_grad,w=w)
            # Compute Batch Hessian with Batch from previous step
            H_b = hess(batch=batch_prev,w=w)
            #H_b = self.batch_fisher(batch=b,w=self.w.copy())
            # Store Hessian for later averaging
            H_list.append(H_b)
            # Convert to array for weighted averaging
            H_to_avg = np.array(H_list)
            # Compute weights for the average
            w_curr = w_list[-1]
            weight = np.exp(-𝜂*np.linalg.norm(x=np.array(w_list)-w_curr,ord=2,axis=1))
            # Compute weighted average of batch Hessian 
            H = np.average(H_to_avg,axis=0,weights=weight)
            # Conditioning matrix C
            C = H + np.power(c/k,power) * Ip
            # descent direction
            d = np.linalg.solve(C,gradient_batch) 
            # update parameter 
            step = lr *(1/(k+k0))**alpha_power
            w -= step * d
            # save current batch for next hessian estimate
            batch_prev = batch
            # average iterate
            w_bar = w_bar + (w-w_bar)/(k+1)
            # Save current iterate
            w_list.append(w_bar.copy())
    return np.array(w_list)
