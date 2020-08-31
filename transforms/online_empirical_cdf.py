import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
class OnlineEmpiricalCDF():
    def __init__(self):
        pass
 
    def get_cont_latent(self, x_batch_obs, window):
        """
        Return the latent variables corresponding to the continuous entries of 
        self.X. Estimates the CDF columnwise with the empyrical CDF
        """
        ecdf = ECDF(window)
        l = len(window)
        return norm.ppf((l / (l + 1.0)) * ecdf(x_batch_obs))

    def get_cont_observed(self, z_batch_missing, window):
        """
        Applies marginal scaling to convert the latent entries in Z corresponding
        to continuous entries to the corresponding imputed oberserved value
        """
        #print(len(z_batch_missing))
        quantiles = norm.cdf(z_batch_missing)
        #print("max quantiles:" +str(max(quantiles)) + "min quantiles:" +str(min(quantiles)))
        return np.quantile(window, quantiles)