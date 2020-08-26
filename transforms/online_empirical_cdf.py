import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
class OnlineEmpiricalCDF():
    def __init__(self, X=None):
        self.X = X
        self.X = None
        if X is not None:
            self.partial_fit(X)

    def partial_fit(self, X_batch):
        if self.X is None:
            self.X = np.copy(X_batch)
        elif X_batch.shape[0] > 0:
            self.X = np.hstack((self.X, X_batch))
        # the previous partial_fit will store all the data it has seen so far
        # self.X = window_at_the_column

    def get_cdf(self, X_batch):
        """
        Return the latent variables corresponding to the continuous entries of 
        self.X. Estimates the CDF columnwise with the empyrical CDF
        """
        X_noNan = self.X[~np.isnan(self.X)]
        ecdf = ECDF(X_noNan)
        n_col = len(X_noNan)
        Z_cont = norm.ppf((n_col / (n_col + 1.0)) * ecdf(X_batch))
        return Z_cont

    def get_inverse_cdf(self, Z_batch):
        """
        Applies marginal scaling to convert the latent entries in Z corresponding
        to continuous entries to the corresponding imputed oberserved value
        """
        X_noNan = self.X[~np.isnan(self.X)]
        X_imp = np.quantile(X_noNan, norm.cdf(Z_batch))
        return X_imp