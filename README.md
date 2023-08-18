# Asymptotic Analysis of Conditioned Stochastic Gradient Descent

This is the code for the article "Asymptotic Analysis of Conditioned Stochastic Gradient Descent", by Rémi Leluc and François Portier, published at Transactions on Machine Learning Research (08/2023) and available on OpenReview at [this url](https://openreview.net/forum?id=U4XgzRjfF1).

## Description

Folders
- graphs/  : contains the graphs from the paper
- results/ :contains the results of the numerical experiments 

Dependencies in Python 3
- requirements.txt : dependencies

Python scripts
- models.py     : implements Ridge and Logistic regression models
- fit_methods.py: implements the (C)-SGD methods
- simus.py      : implements simulation routines for Ridge and Logistic regression models
- ridge.py      : runs simulation routines for Ridge regression models and saves results

Notebook:
- plot_results.ipynb : Load the results and plot the Figures of the paper

## Citation

If you refer to this work, please cite as

```
@article{
leluc2023asymptotic,
title={Asymptotic Analysis of Conditioned Stochastic Gradient Descent},
author={R{\'e}mi Leluc and Fran{\c{c}}ois Portier},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=U4XgzRjfF1},
note={}
}

