import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
class OnlineOrdinalMarginalEstimator():
    def __init__(self):
        pass


    def get_ord_latent(self, x_batch_obs, window):
        """
        get the cdf at each point in X_batch
        """
        # the lower endpoint of the interval for the cdf
        ecdf = ECDF(window)
        unique = np.unique(window)
        threshold = np.min(np.abs(unique[1:] - unique[:-1]))/2.0
        z_lower_obs = norm.ppf(ecdf(x_batch_obs - threshold))
        z_upper_obs = norm.ppf(ecdf(x_batch_obs + threshold))
        return z_lower_obs, z_upper_obs


    def get_ord_observed(self, z_batch_missing, window, DECIMAL_PRECISION = 3):
        """
        Gets the inverse CDF of Q_batch
        returns: the Q_batch quantiles of the ordinals seen thus far
        """
        n = len(window)
        x = norm.cdf(z_batch_missing)
        # round to avoid numerical errors in ceiling function
        quantile_indices = np.ceil(np.round_((n + 1) * x - 1, DECIMAL_PRECISION))
        quantile_indices = np.clip(quantile_indices, a_min=0,a_max=n-1).astype(int)
        sort = np.sort(window)
        return sort[quantile_indices]



        