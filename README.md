# GaussianCopulaImp
This package provides a python implemention to fit a Gaussian copula model, on continuous, ordinal and binary mixed data with missing values. The user could either fit a full rank Gaussian copula model [1] or a low rank Gaussian copula model [2], which builds upon the PPCA model in a latent space. The package also includes the mini-batch and online implementation of the full rank Gaussian copula model [3], which can accelerate the training process and adapt to a changing distribution in the streaming data. The fitted model can be used for missing value imputation, latent correlation learning and latent correlation change detection.

## Installation

The easiest way is to install using pip: `pip install GaussianCopulaImp` 

## Overview

There are two different models available for use: the standard Gaussian copula model and the low rank Gaussian copula model. When working with skinny datasets (small p), the standard Gaussian copula model is preferred. In contrast, the low rank Gaussian copula model is recommended when working with wide datasets (large p). 

There are three training options for the standard Gaussian copula model: standard offline training, mini-batch offline training and mini-batch online training. In short, mini-batch offline training is often much faster than standard offline training, by using more frequent model updates. Online training is designed for the streaming data scenario when data comes  at different time points. Parallelism is now supported for all training options with the standard Gaussian copula model. 

For low rank Gaussian copula model, only standard offline training without parallelism is supported at this moment. Parallelism will be supported soon. The development of mini-batch training (both offline and online) are nontrivial. Please contact the authors if you are interested in collaboration for developing those functionalities.

Please also see below for more detailed dicussions on how to select the model and training option that works best for your purpose.

## Examples 
```
from GaussianCopulaImp.expectation_maximization import ExpectationMaximization as EM
import numpy as np

# generate 2-dim mixed data by monotonically transforming 2-dim Gaussian
np.random.seed(101)
X = np.random.multivariate_normal(mean = [0, 0], cov = [[1, 0.7], [0.7, 1]], size = 500)
X[:,1] = np.digitize(X[:,1], [-2, -1, 0, 1, 2])

# randomly remove 30% entries in each column but avoid an empty row
mask_size_each = int(500*0.3)
mask_rows_id = np.random.choice(np.arange(500), size=mask_size_each*2)
X_mask = X.copy()
X_mask[mask_rows_id[:mask_size_each], 0] = np.nan
X_mask[mask_rows_id[mask_size_each:], 1] = np.nan

# model fitting 
em = EM()
X_imp, sigma_est = em.impute_missing(X=X_mask, verbose=True)

# Evaluation
print(f'Estimated latent correlation is {sigma_est[0,1]}')
err_cont = X_imp[mask_rows_id[:mask_size_each], 0] - X[mask_rows_id[:mask_size_each], 0]
nrmse_cont = np.sqrt(np.power(err_cont, 2).mean()/np.power(X[mask_rows_id[:mask_size_each], 0],2).mean())
err_ord = X_imp[mask_rows_id[mask_size_each:], 1] - X[mask_rows_id[mask_size_each:], 1]
mae_ord = np.abs(err_ord).mean()
print(f'Imputation error: \n NRMSE for the continuous variable is {nrmse_cont:.3f} \n MAE for the ordinal variable is {mae_ord:.3f}')
```



## References
[1] Zhao, Y. and Udell, M. Missing value imputation for mixed data via Gaussian copula, KDD 2020.

[2] Zhao, Y. and Udell, M. Matrix completion with quantified uncertainty through low rank Gaussian copula, NeurIPS 2020.

[3] Zhao, Y., Landgrebe, E., Shekhtman E., and Udell, M. Online missing value imputation and correlation change detection for mixed-type Data via Gaussian Copula, arXiv 2020.
