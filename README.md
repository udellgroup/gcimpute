# online_mixed_gc_imp
This package provides a python implemention of online, minibatch, and parallel implementations of Gaussian Copula Imputation.

## ExpectationMaximization
Implements an approximate expectation maximization algorithm for fitting a Guassian Copula to impute missing values that supports parallelism as in [1]. An example of usage can be found in evaluation/copula_generate/standard_copula_generated.py

## BatchExpectationMaximization
Implements an approximate minibatch expectation maximization algorithm for fitting a Guassian Copula to impute missing values that supports parallelism within minibatches as in [2]. An example of usage can be found in evaluation/copula_generate/batch_copula_generated.py

## OnlineExpectationMaximization
Implements an approximate online expectation maximization algorithm for fitting a Guassian Copula to impute missing values that supports parallelism within minibatches as in [2]. An example of usage can be found in evaluation/copula_generate/online_copula_generated.py


## References
[1] Zhao, Y. and Udell, M. Missing value imputation for mixeddata through gaussian copula, 2019.

[2] Landgrebe, E., Zhao, Y., and Udell, M. Online Mixed Missing Value Imputation Using Gaussian Copula, 2020.