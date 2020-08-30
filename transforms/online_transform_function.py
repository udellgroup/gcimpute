import numpy as np
from scipy.stats import norm
from .online_empirical_cdf import OnlineEmpiricalCDF
from .online_ordinal_marginal_estimator import OnlineOrdinalMarginalEstimator


class OnlineTransformFunction():
    def __init__(self, cont_indices, ord_indices, X=None, window_size=100):
        """
        Require window_size to be positive integers.

        To initialize the window, 
        for continuous columns, sample standatd normal with mean and variance determined by the first batch of observation;
        for ordinal columns, sample uniformly with replacement among all integers between min and max seen from data.

        """
        self.cont_online_marginals = [OnlineEmpiricalCDF() for _ in range(np.sum(cont_indices))]
        self.ord_online_marginals = [OnlineOrdinalMarginalEstimator() for _ in range(np.sum(ord_indices))]
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        p = len(cont_indices)
        self.window_size = window_size
        self.window = np.array([[None for x in range(p)] for y in range(self.window_size)]).astype(np.float64)
        self.update_pos = np.zeros(p).astype(np.int)
        self.batch_dim = None
        if X is not None:
            self.partial_fit(X)
        
    def partial_fit(self, X_batch):
        """
        Update the marginal estimate with the data in X
        """
        if self.window[0, 0] is None:
            # YX comment: improve the initial sampling as described in lines 12-14
            # the initial sampling should be done column-wisely
            self.window[:, self.cont_indices] = 0
            self.window[:, self.ord_indices] = min(X_batch[:, self.ord_indices])
        for row in X_batch:
            for col_num in range(len(row)):
                data = row[col_num]
                if not np.isnan(data):
                    self.window[self.update_pos[col_num], col_num] = data
                    self.update_pos[col_num] += 1 
                    if self.update_pos[col_num] >= self.window_size:
                        self.update_pos[col_num] = 0
        cont_entries = self.window[:, self.cont_indices]
        ord_entries = self.window[:, self.ord_indices]
        self.batch_dim = X_batch.shape
        # IT SUFFICES TO UPDATE THE WINDOW WHEN NEW DATA POINTS COME IN 
        # update all the continuos marginal estimates
        #for cont_entry, cont_online_marginal in zip(cont_entries.T, self.cont_online_marginals):
         #   cont_online_marginal.partial_fit(cont_entry[~np.isnan(cont_entry)]) 

        # update all the ordinal marginal estimates
        for ord_entry, ord_online_marginal in zip(ord_entries.T, self.ord_online_marginals):
            ord_online_marginal.partial_fit(ord_entry[~np.isnan(ord_entry)])

    def partial_evaluate_cont_latent(self, X_batch):
        """
        Obtain the latent continuous values corresponding to X_batch 
        """
        X_cont = X_batch[:,self.cont_indices]
        window_cont = self.window[:,self.cont_indices]
        Z_cont = np.copy(X_cont)
        for i,cont_online_marginal in enumerate(self.cont_online_marginals):
            # INPUT THE WINDOW FOR EVERY COLUMN
            missing = np.isnan(X_cont[:,i])
            Z_cont[~missing,i] = cont_online_marginal.get_cdf(X_cont[~missing,i], window_cont[:,i])
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
            missing = np.isnan(ord_entry)
            # INPUT THE WINDOW FOR EVERY COLUMN
            Z_ord_lower[~missing,i], Z_ord_upper[~missing,i] = ord_online_marginal.get_cdf(ord_entry[~missing])
            # clip to prevent errors due to numerical imprecisions of floating poitns
            Z_ord_lower[~missing,i] = norm.ppf(np.clip(Z_ord_lower, a_min=0, a_max=1)[~missing,i])
            Z_ord_upper[~missing,i] = norm.ppf(np.clip(Z_ord_upper, a_min=0, a_max=1)[~missing,i])
        return Z_ord_lower, Z_ord_upper

    def partial_evaluate_cont_observed(self, Z_batch, X_batch):
        """
        Transform the latent continous variables in Z_batch into corresponding observations
        """
        Z_cont = Z_batch[:,self.cont_indices]
        X_cont = X_batch[:,self.cont_indices]
        window_cont = self.window[:,self.cont_indices]
        for i,cont_online_marginal in enumerate(self.cont_online_marginals):
            missing = np.isnan(X_cont[:,i])
            X_cont[missing,i] = cont_online_marginal.get_inverse_cdf(Z_cont[missing,i], window_cont[:,i])
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

        
