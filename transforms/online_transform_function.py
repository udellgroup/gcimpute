import numpy as np
from scipy.stats import norm
from .online_empirical_cdf import OnlineEmpiricalCDF
from .online_ordinal_marginal_estimator import OnlineOrdinalMarginalEstimator


class OnlineTransformFunction():
    def __init__(self, cont_indices, ord_indices, X=None):
        self.cont_online_marginals = [OnlineEmpiricalCDF() for _ in range(np.sum(cont_indices))]
        self.ord_online_marginals = [OnlineOrdinalMarginalEstimator() for _ in range(np.sum(ord_indices))]
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        if X is not None:
            self.partial_fit(X)
        
    def partial_fit(self, X_batch):
        """
        Update the marginal estimate with the data in X
        """
        cont_entries = X_batch[:, self.cont_indices]
        ord_entries = X_batch[:, self.ord_indices]
        # update all the continuos marginal estimates
        for cont_entry, cont_online_marginal in zip(cont_entries.T, self.cont_online_marginals):
            cont_online_marginal.partial_fit(cont_entry[~np.isnan(cont_entry)])
        # update all the ordinal marginal estimates
        for ord_entry, ord_online_marginal in zip(ord_entries.T, self.ord_online_marginals):
            ord_online_marginal.partial_fit(ord_entry[~np.isnan(ord_entry)])

    def partial_evaluate_cont_latent(self, X_batch):
        """
        Obtain the latent continuous values corresponding to X_batch 
        """
        cont_batch = X_batch[:,self.cont_indices]
        Z_cont = np.empty(cont_batch.shape)
        Z_cont[:] = np.nan
        for i,(cont_entry, cont_online_marginal) in enumerate(zip(cont_batch.T, self.cont_online_marginals)):
            Z_cont[~np.isnan(cont_entry),i] = cont_online_marginal.get_cdf(cont_entry[~np.isnan(cont_entry)])
        return Z_cont

    def partial_evaluate_ord_latent(self, X_batch):
        """
        Obtain the latent ordinal values corresponding to X_batch
        """
        ord_batch = X_batch[:,self.ord_indices]
        Z_ord_lower = np.empty(ord_batch.shape)
        Z_ord_lower[:] = np.nan
        Z_ord_upper = np.empty(ord_batch.shape)
        Z_ord_upper[:] = np.nan
        for i,(ord_entry, ord_online_marginal) in enumerate(zip(ord_batch.T, self.ord_online_marginals)):
            Z_ord_lower[~np.isnan(ord_entry),i], Z_ord_upper[~np.isnan(ord_entry),i] = ord_online_marginal.get_cdf(ord_entry[~np.isnan(ord_entry)])
            # clip to prevent errors due to numerical imprecisions of floating poitns
            Z_ord_lower[~np.isnan(ord_entry),i] = norm.ppf(np.clip(Z_ord_lower, a_min=0, a_max=1)[~np.isnan(ord_entry),i])
            Z_ord_upper[~np.isnan(ord_entry),i] = norm.ppf(np.clip(Z_ord_upper, a_min=0, a_max=1)[~np.isnan(ord_entry),i])
        return Z_ord_lower, Z_ord_upper

    def partial_evaluate_cont_observed(self, Z_batch):
        """
        Transform the latent continous variables in Z_batch into corresponding observations
        """
        Z_batch_cont = Z_batch[:,self.cont_indices]
        X_cont = np.empty(Z_batch_cont.shape)
        for i,(cont_entry, cont_online_marginal) in enumerate(zip(Z_batch_cont.T, self.cont_online_marginals)):
            X_cont[:,i] = cont_online_marginal.get_inverse_cdf(cont_entry)
        return X_cont

    def partial_evaluate_ord_observed(self, Z_batch):
        """
        Transform the latent ordinal variables in Z_batch into corresponding observations
        """
        Z_batch_ord = Z_batch[:,self.ord_indices]
        X_ord = np.empty(Z_batch_ord.shape)
        for i,(ord_entry, ord_online_marginal) in enumerate(zip(Z_batch_ord.T, self.ord_online_marginals)):
            X_ord[:,i] = ord_online_marginal.get_inverse_cdf(norm.cdf(ord_entry))
        return X_ord

        
