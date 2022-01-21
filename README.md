# gcimpute
This package provides a python implemention to fit a Gaussian copula model, on continuous, ordinal and binary mixed data with missing values. The user could either fit a full rank Gaussian copula model [1] or a low rank Gaussian copula model [2], which builds upon the PPCA model in a latent space. The package also includes the mini-batch and online implementation of the full rank Gaussian copula model [3], which can accelerate the training process and adapt to a changing distribution in the streaming data. The fitted model can be used for missing value imputation, latent correlation learning and latent correlation change detection.

## Installation

The easiest way is to install using pip: 

`
pip install gcimpute
` 

If you want to customize the source code, you may install in the editable mode by first `git clone` this respository, and then do

`
pip install -e .
`

in the cloned directory.

## Overview

There are two different models available for use: the standard Gaussian copula model and the low rank Gaussian copula model. When working with skinny datasets (small p), the standard Gaussian copula model is preferred. In contrast, the low rank Gaussian copula model is recommended when working with wide datasets (large p). 

There are three training options for the standard Gaussian copula model: standard offline training, mini-batch offline training and mini-batch online training. In short, mini-batch offline training is often much faster than the standard offline training, by using more frequent model updates. Online training is designed for the streaming data scenario when data comes  at different time points or the data distribution is changing over time. Parallelism is now supported for all training options with the standard Gaussian copula model for further acceleration. 

For low rank Gaussian copula model, only standard offline training with/without parallelism is supported. The development of mini-batch training (both offline and online) is nontrivial. Please contact the authors if you are interested in collaboration for developing those functionalities.

Please also see below for more detailed dicussions on how to select the model and training option that works best for your purpose.

## Select the right model to use
Both the standard Gaussian copula model (GC for short) and the low rank Gaussian copula model (LRGC for short) estimate a copula correlation matrix of size p by p for p variables. The difference is that LRGC imposes a low rank structure on the latent correlation matrix such that the number of free parameters is O(pk), with a specified low rank k. In contrast, GC has O(p^2) free parameters.

GC is recommended in the classical setting where the number of samples is much larger than the number of variabels, i.e. n>>p. 
For high-dimensional setting (p>n or large p), LRGC is recommended because (1) the estimated copula correlation matrix will be singular when p>n; (2) the O(np^3) time complexity of GC may be too expensive for large p, while LRGC only has O(npk^2) time complexity with a small specified rank k.

Most of our experiments have at least around 200 samples points (n>200). There may be some problematic behavior when n is very small, since the copula marginal estimation accuracy depends on n. Please report any bugs you encounter!

## Select the right training option
The **standard offline training** is an approximate EM algorithm, the simplest one to use. **Mini-batch offline training** is an online EM algorithm, which can greatly accelerate the training by more frequent model updates. **Mini-batch online training** differs with **mini-batch offline training** in three places: (1) How many data points they remember. The offline version remebers all the data to improve the copula marginal estimation accuracy, while the online version only remembers recent data in a lookback window (length determined by the user); (2) How many passes over the data are allowed. The offline version can take as many passes as needed, while the online version can only take one pass; (3) Whether the order of data rows can be randomly shuffled. The offline version can shuffle the data at each pass, while the online version does not shuffle the data during fitting.

When the computation time is the only consideration, **mini-batch offline training** is prefered. However, there are situations that remembering all data points is either impossible or undesirable. For example, when the data comes at different times and a model is required at each arrival time, future data is unavailable and remembering all data is thus impossible. Also, when the data distribution changes over time, remebering all data points includes much noise in historical data that no longer helps explaining current data and thus undesirable. For these situations, **mini-batch online training** should be used. See Examples/Example Get Started.ipynb for how to correctly use each training option.

When using **minibatch offline training**, we find using decaying learning rate c/(c+k) at the k-th update [3] works well in most cases. We mostly tune `c` in the range 1~10 and find a value 5 in general works well. The value of  `batch_size` should be determined with `max_iter` together. Often we first determine the value of `batch_size` and then determine `max_iter` by the number of passes desired.

when using **minibatch online training**, one can either use a constant learning rate in the range (0,1) or a decaying learning rate as when using **minibatch offline training**. A constant learning rate should be used if the data is believed to change over time. Our default choice 0.5 works well in most cases. Another parameter `window_size` is used to control how many data points to remember. It is not used for minibatch offline training, because all data points are remembered in the minibatch offline training. The default value is 200. The general suggestion for tuning `window_size` is to use large values if the data distribution is stable but small values if the data distribution is changing. 

For both **minibatch offline training** and **minibatch online training**, the general suggestion on the value of `batch_size` is to use a value larger than the number of variables, for the sake of numerical stability and theoretical guarantee. We usually do not tune on `batch_size` much and the default value `batch_size=100` works well in most of our experiments.

## Examples 
```python
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.helper_data import generate_mixed_from_gc
from gcimpute.helper_evaluation import get_smae
from gcimpute.helper_mask import mask_types
import numpy as np
seed = 101

# generate and mask 15-dim mixed data (5 continuous variables, 5 ordinal variables (1-5) and 5 boolean variables) 
X = generate_mixed_from_gc(n=2000, seed=seed)
X_mask = mask_types(X, mask_num=2, seed=seed)
print('The first row of the masked dataset: ')
print(X_mask[0,:])

# model fitting 
model = GaussianCopula(verbose=1)
X_imp = model.fit_transform(X=X_mask)

# Evaluation: compute the scaled-MAE (SMAE) for each data type (scaled by MAE of median imputation) 
smae = get_smae(X_imp, X, X_mask)
print(f'The SMAE across continous variables: mean {smae[:5].mean():.3f} and std {smae[:5].std():.3f}')
print(f'The SMAE across oridnal variables: mean {smae[5:10].mean():.3f} and std {smae[5:10].std():.3f}')
print(f'The SMAE across boolean variables: mean {smae[10:].mean():.3f} and std {smae[10:].std():.3f}')
```

More detailed examples are available under directory Examples.

## References
[1] Zhao, Y. and Udell, M. Missing value imputation for mixed data via Gaussian copula, KDD 2020.

[2] Zhao, Y. and Udell, M. Matrix completion with quantified uncertainty through low rank Gaussian copula, NeurIPS 2020.

[3] Zhao, Y., Landgrebe, E., Shekhtman E., and Udell, M. Online missing value imputation and correlation change detection for mixed-type Data via Gaussian Copula, arXiv 2020.
