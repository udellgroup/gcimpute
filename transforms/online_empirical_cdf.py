import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
class OnlineEmpiricalCDF():
    def __init__(self):
        pass
 
    def get_cdf(self, x_batch, window, scaling=1000):
        """
        Return the latent variables corresponding to the continuous entries of 
        self.X. Estimates the CDF columnwise with the empyrical CDF
        """
        ecdf = ECDF(window)
        return norm.ppf((scaling / (scaling + 1.0)) * ecdf(x_batch))

    def get_inverse_cdf(self, z_batch, window):
        """
        Applies marginal scaling to convert the latent entries in Z corresponding
        to continuous entries to the corresponding imputed oberserved value
        """
        return np.quantile(window, norm.cdf(z_batch))