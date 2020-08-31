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
        self.window = np.array([[np.nan for x in range(p)] for y in range(self.window_size)]).astype(np.float64)
        #print(self.window.shape)
        self.update_pos = np.zeros(p).astype(np.int)
        if X is not None:
            self.partial_fit(X)
        
    def partial_fit(self, X_batch):
        """
        Update the running window used to estimate marginals with the data in X
        """
        if np.isnan(self.window[0, 0] ):
            # YX comment: improve the initial sampling as described in lines 12-14
            # the initial sampling should be done column-wisely
            mean_ord = np.nanmean(X_batch[:, self.cont_indices])
            std_ord = np.nanstd(X_batch[:, self.cont_indices])
            self.window[:, self.cont_indices] = np.random.normal(mean_ord, std_ord, size=(self.window_size, np.sum(self.cont_indices)))
            # ord_values = np.unique(X_batch[:, self.ord_indices]) # np.fromfunction(lambda i,j: np.random.choice(ord_values), size=self.window[:, self.ord_indices].shape)
            #min_ord = min(X_batch[:, self.ord_indices])
            #max_ord = max(X_batch[:, self.ord_indices]) + 1
            #self.window[:, self.ord_indices] = np.random.randint(min_ord, max_ord, size=self.window[:, self.ord_indices].shape)
            for j,loc in enumerate(self.ord_indices):
                if loc:
                    min_ord = np.nanmin(X_batch[:, j])
                    max_ord = np.nanmax(X_batch[:,j]) + 1
                    self.window[:, j] = np.random.randint(min_ord, max_ord, size=self.window_size)
        #print(np.sum(np.isnan(self.window)))
        #print(np.sum(np.isnan(self.window), 0))
        for row in X_batch:
            for col_num in range(len(row)):
                data = row[col_num]
                if not np.isnan(data):
                    self.window[self.update_pos[col_num], col_num] = data
                    self.update_pos[col_num] += 1 
                    if self.update_pos[col_num] >= self.window_size:
                        self.update_pos[col_num] = 0
        # IT SUFFICES TO UPDATE THE WINDOW WHEN NEW DATA POINTS COME IN 
        #cont_entries = self.window[:, self.cont_indices]
        # update all the continuos marginal estimates
        #for cont_entry, cont_online_marginal in zip(cont_entries.T, self.cont_online_marginals):
         #   cont_online_marginal.partial_fit(cont_entry[~np.isnan(cont_entry)]) 

        #ord_entries = self.window[:, self.ord_indices]
        # update all the ordinal marginal estimates
        #for ord_entry, ord_online_marginal in zip(ord_entries.T, self.ord_online_marginals):
        #    ord_online_marginal.partial_fit(ord_entry[~np.isnan(ord_entry)])
        #print(np.sum(np.isnan(self.window)))
        #print(np.sum(np.isnan(self.window), 0))

    def partial_evaluate_cont_latent(self, X_batch):
        """
        Obtain the latent continuous values corresponding to X_batch 
        """
        X_cont = X_batch[:,self.cont_indices]
        window_cont = self.window[:,self.cont_indices]
        Z_cont = np.empty(X_cont.shape)
        Z_cont[:] = np.nan
        for i,cont_online_marginal in enumerate(self.cont_online_marginals):
            # INPUT THE WINDOW FOR EVERY COLUMN
            missing = np.isnan(X_cont[:,i])
            Z_cont[~missing,i] = cont_online_marginal.get_cont_latent(X_cont[~missing,i], window_cont[:,i])
        return Z_cont

    def partial_evaluate_ord_latent(self, X_batch):
        """
        Obtain the latent ordinal values corresponding to X_batch
        """
        X_ord = X_batch[:,self.ord_indices]
        window_ord = self.window[:,self.ord_indices]
        Z_ord_lower = np.empty(X_ord.shape)
        Z_ord_lower[:] = np.nan
        Z_ord_upper = np.empty(X_ord.shape)
        Z_ord_upper[:] = np.nan
        for i, ord_online_marginal in enumerate(self.ord_online_marginals):
            missing = np.isnan(X_ord[:,i])
            # INPUT THE WINDOW FOR EVERY COLUMN
            Z_ord_lower[~missing,i], Z_ord_upper[~missing,i] = ord_online_marginal.get_ord_latent(X_ord[~missing,i], window_ord[:,i])
        return Z_ord_lower, Z_ord_upper

    def partial_evaluate_cont_observed(self, Z_batch, X_batch):
        """
        Transform the latent continous variables in Z_batch into corresponding observations
        """
        Z_cont = Z_batch[:,self.cont_indices]
        X_cont = X_batch[:,self.cont_indices]
        X_cont_imp = np.copy(X_cont)
        window_cont = self.window[:,self.cont_indices]
        for i,cont_online_marginal in enumerate(self.cont_online_marginals):
            missing = np.isnan(X_cont[:,i])
            ##print("length of missing : " +str(sum(missing)) + " at cont col "+str(i))
            X_cont_imp[missing,i] = cont_online_marginal.get_cont_observed(Z_cont[missing,i], window_cont[:,i])
        return X_cont_imp

    def partial_evaluate_ord_observed(self, Z_batch, X_batch):
        """
        Transform the latent ordinal variables in Z_batch into corresponding observations
        """
        Z_ord = Z_batch[:,self.ord_indices]
        X_ord = X_batch[:, self.ord_indices]
        X_ord_imp = np.copy(X_ord)
        window_ord = self.window[:,self.ord_indices]
        for i, ord_online_marginal in enumerate(self.ord_online_marginals):
            missing = np.isnan(X_ord[:,i])
            X_ord_imp[missing,i] = ord_online_marginal.get_ord_observed(Z_ord[missing,i], window_ord[:,i])
        return X_ord_imp



        
