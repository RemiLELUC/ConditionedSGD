
'''
This file runs simulations routines 
for Ridge regression models and save results.
Authors: Rémi Leluc, François Portier
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from models import linearReg
from simus import run_ols
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm


### Simulate data for Ridge regression
# Number of samples and dimension
n_samples = 10000   # number of samples
n_features = 20     # dimension of the problem
#n_features = 100   # dimension of the problem

# Simulate data for regression
seed=0
noise=1
X,y = make_regression(n_samples=n_samples,
                     n_features=n_features,
                      noise=noise,random_state=seed)
y/=y.sum()


### Compute optimal parameter
𝜆 = 1/n_samples          #regularization parameter
G = ((X.T)@X)/n_samples  # Gram matrix
A = G + 𝜆*np.eye(n_features)
B = ((X.T)@y)/n_samples
# compute ridge solution
ridge = np.linalg.solve(a=A ,b=B)
data_opt = np.sum((y-X.dot(ridge))**2)/(2*n_samples)
reg_opt = (𝜆/2) * sum(ridge**2)
loss_opt = data_opt + reg_opt
print('data_opt:',data_opt)
print('reg_opt :',reg_opt)
print(loss_opt)


### Parameter simulations
N = int(1e2)    # number of passes over coordinates
N_exp = 100     # number of experiments
lr = 2          # numerator of learning rate
k0 = 5          # smoothing parameter, denominator of learning rate
alpha_power = 1 # power in the learning rate
batch_size= 16  # batch size for gradient estimates 
batch_hess= 16  # batch size for hessian estimates
c=1             # numerator learning rate gamma_k for C_k
power =1/2      # power of learning rate gamma_k for C_k
𝜂 = 200         # importance weights parameter
same=True       # same batch for gradient/hessian
burn_in = 30    # burn in period for Polyak averaging


# Sgd and Polyak variant
# Standard SGD
w_sgd,loss_sgd = run_ols(X=X,y=y,𝜆=𝜆,method='sgd',N_exp=N_exp,N=N,burn_in=burn_in,
                         batch_size=batch_size,batch_hess=batch_hess,lr=lr,c=c,k0=k0,a=1,
                         alpha_power=alpha_power,𝜂=None,power=None,same=None)
mean_sgd = np.mean(loss_sgd,axis=0)

# Polyak averaging
k0 = 6
burn_in=15
w_sgd_avg,loss_sgd_avg = run_ols(X=X,y=y,𝜆=𝜆,method='sgd-avg',N_exp=N_exp,N=N,burn_in=burn_in,
                                 batch_size=batch_size,batch_hess=batch_hess,lr=lr,c=c,k0=k0,a=1,
                                 alpha_power=2/3,𝜂=None,power=None,same=None)
mean_sgd_avg = np.mean(loss_sgd_avg,axis=0)

# Conditioned Sgd with equal and adaptive weights
# C-SGD equal
lr=3
k0 = 6
w_equal,loss_equal = run_ols(X=X,y=y,𝜆=𝜆,method='csgd',N_exp=N_exp,N=N,burn_in=burn_in,
                             batch_size=batch_size,batch_hess=batch_hess,lr=lr,c=c,k0=k0,a=1,
                             alpha_power=alpha_power,𝜂=0,power=power,same=same)
mean_equal = np.mean(loss_equal,axis=0)
# C-SGD weighted
lr = 8
k0 = 20
w_weighted,loss_weighted = run_ols(X=X,y=y,𝜆=𝜆,method='csgd',N_exp=N_exp,N=N,burn_in=burn_in,
                                   batch_size=batch_size,batch_hess=batch_hess,lr=lr,c=c,k0=k0,a=1,
                                   alpha_power=alpha_power,𝜂=10,power=power,same=same)
mean_weighted = np.mean(loss_weighted,axis=0)

#AdaFull-avg
# AdaFul avg
a = 0.3
lr = 0.5
k0 = 3
w_adadiag_avg,loss_adadiag_avg = run_ols(X=X,y=y,𝜆=𝜆,method='adafull-avg',N_exp=N_exp,N=N,burn_in=burn_in,
                         batch_size=batch_size,batch_hess=batch_hess,lr=lr,c=c,k0=k0,a=a,
                         alpha_power=alpha_power,𝜂=None,power=None,same=None)
mean_adadiag_avg = np.mean(loss_adadiag_avg,axis=0)

### Save results

#np.save('ridge_sgd_d20.npy',(mean_sgd-loss_opt)/(mean_sgd[0]-loss_opt))
#np.save('ridge_sgd_avg_d20.npy',(mean_sgd_avg-loss_opt)/(mean_sgd_avg[0]-loss_opt))
#np.save('ridge_equal_d20.npy',(mean_equal-loss_opt)/(mean_equal[0]-loss_opt))
#np.save('ridge_weighted_d20.npy',(mean_weighted-loss_opt)/(mean_weighted[0]-loss_opt))
#np.save('ridge_adafull_avg_d20.npy',(mean_adadiag_avg-loss_opt)/(mean_adadiag_avg[0]-loss_opt))
