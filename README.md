# online_mixed_gc_imp
This package provides a python implemention to fit a full rank Gaussian copula model [1] or low rank Gaussian copula model [2], on continuous, ordinal and binary mixed data with missing values. The package also includes the mini-batch and online implementation of the full rank Gaussian copula model. The fitted model can be used for missing value imputation, correlation structure learning and correlation change detection [1,2,3].

# Installation

To install this package, clone the repo and run `python3 setup.py install`.

## ExpectationMaximization
Implements an approximate expectation maximization algorithm for fitting a Guassian Copula to impute missing values that supports parallelism as in [1]. An example of usage can be found in evaluation/copula_generate/standard_copula_generated.py. 

# LowRankExpectationMaximization
Implements an approximate expectation maximization algorithm for fitting a low rank Gaussian Guassian Copula to impute missing values as in [2]. Examples of usage can be found in evaluation/LRGC. 


## BatchExpectationMaximization
Implements an approximate minibatch expectation maximization algorithm for fitting a Guassian Copula to impute missing values that supports parallelism within minibatches as in [3]. An example of usage can be found in evaluation/copula_generate/batch_copula_generated.py

## OnlineExpectationMaximization
Implements an approximate online expectation maximization algorithm for fitting a Guassian Copula to impute missing values that supports parallelism within minibatches as in [3]. An example of usage can be found in evaluation/copula_generate/online_copula_generated.py



## References
[1] Zhao, Y. and Udell, M. Missing value imputation for mixeddata through gaussian copula, KDD 2020.

[2] Zhao, Y. and Udell, M. Matrix completion with quantified uncertainty through low rank Gaussian copula, NeurIPS 2020.

[3] Zhao, Y., Landgrebe, E., Shekhtman E., and Udell, M. Online missing value imputation and correlation change detection for mixed-type Data via Gaussian Copula, arXiv 2020.
