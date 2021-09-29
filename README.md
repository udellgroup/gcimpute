# GaussianCopulaImp
This package provides a python implemention to fit a Gaussian copula model, on continuous, ordinal and binary mixed data with missing values. The user could either fit a full rank Gaussian copula model [1] or a low rank Gaussian copula model [2], which builds upon the PPCA model in a latent space. The package also includes the mini-batch and online implementation of the full rank Gaussian copula model [3], which can accelerate the training process and adapt to a changing distribution in the streaming data. The fitted model can be used for missing value imputation, latent correlation learning and latent correlation change detection.

## Installation

The easiest way is to install using pip: `pip install GaussianCopulaImp` 

## Overview

There are two different models available for use: the standard Gaussian copula model and the low rank Gaussian copula model. When working with skinny datasets (small p), the standard Gaussian copula model is preferred. In contrast, the low rank Gaussian copula model is recommended when working with wide datasets (large p). 

There are three training options for the standard Gaussian copula model: standard offline training, mini-batch offline training and mini-batch online training. In short, mini-batch offline training is often much faster than the standard offline training, by using more frequent model updates. Online training is designed for the streaming data scenario when data comes  at different time points or the data distribution is changing over time. Parallelism is now supported for all training options with the standard Gaussian copula model for further acceleration. 

For low rank Gaussian copula model, only standard offline training without parallelism is supported at this moment. Parallelism will be supported soon. The development of mini-batch training (both offline and online) is nontrivial. Please contact the authors if you are interested in collaboration for developing those functionalities.

Please also see below for more detailed dicussions on how to select the model and training option that works best for your purpose.

## Select the right model to use
Both the standard Gaussian copula model (GC for short) and the low rank Gaussian copula model (LRGC for short) estimate a copula correlation matrix of size p by p for p variables. The difference is that LRGC imposes a low rank structure on the latent correlation matrix such that the number of free parameters is O(pk), with a specified low rank k. In contrast, GC has O(p^2) free parameters.

GC is recommended in the classical setting where the number of samples is much larger than the number of variabels, i.e. n>>p. 
For high-dimensional setting (p>n or large p), LRGC is recommended because (1) the estimated copula correlation matrix will be singular when p>n; (2) the O(np^3) time complexity of GC may be too expensive for large p, while LRGC only has O(npk^2) time complexity with a small specified rank k.

Most of our experiments have at least around 200 samples points (n>200). There may be some problematic behavior when n is very small, since the copula marginal estimation accuracy depends on n. Please report any bugs you encounter!

## Select the right training option
The standard offline training is an approximate EM algorithm, the simplest one to use. Mini-batch offline training is an online EM algorithm, which can greatly accelerate the training. Mini-batch online training differs with mini-batch offline training only in how many data points they remember. The offline version remebers all the data to improve the copula marginal estimation accuracy, while the online version only remembers recent data in a lookback window (length determined by the user).

When the computation time is the only consideration, mini-batch offline training is prefered. However, there are situations that remembering all data points is either impossible or undesirable. For example, when the data comes at different times and a model is required at each arrival time, future data is unavailable and remembering all data is thus impossible. Also, when the data distribution changes over time, remebering all data points includes much noise in historical data that no longer helps explaining current data and thus undesirable. For these situations, mini-batch online training should be used.

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
