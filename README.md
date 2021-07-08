# Asymptotic Optimality of Conditioned Stochastic Gradient Descent

This is the code for the article "Asymptotic Optimality of Conditioned Stochastic Gradient Descent", Rémi LELUC and François PORTIER.

## Description

Folders
- datasets_LIBSVM/ : contains the mushroom and phishing datasets
- results/         : contains the results of the numerical experiments for simulated and real data.

Dependencies in Python 3
- requirements.txt : dependencies

Python scripts
- models.py     : implements Ridge and Logistic regression models
- fit_methods.py: implements the (C)-SGD methods
- simus.py      : implements simulation routines for Ridge and Logistic regression models

Python notebooks:
- Ridge_simulated_data.ipynb : Perform comparison of methods for simulated data with Ridge regression
- Ridge_real_data.ipynb      : Perform comparison of methods for real data with Ridge regression

- Logistic_simulated_data.ipynb: Perform comparison of methods for simulated data with Logistic regression
- Logistic_real_data.ipynb     : Perform comparison of methods for real data with Logistic regression

- Plot_simulated_data.ipynb: Load the results and plot the Figures of the paper for simulated data
- Plot_real_data.ipynb     : Load the results and plot the Figures of the paper for real data

