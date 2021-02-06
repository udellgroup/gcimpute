import numpy as np
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF


class OnlineTransformFunction():
    def __init__(self, cont_indices, ord_indices, X=None, window_size=100):
        """
        Require window_size to be positive integers.

        To initialize the window, 
        for continuous columns, sample standatd normal with mean and variance determined by the first batch of observation;
        for ordinal columns, sample uniformly with replacement among all integers between min and max seen from data.

        """
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        p = len(cont_indices)
        self.window_size = window_size
        self.window = np.array([[np.nan for x in range(p)] for y in range(self.window_size)]).astype(np.float64)
        self.update_pos = np.zeros(p).astype(np.int)
        if X is not None:
            self.partial_fit(X)
        
    def partial_fit(self, X_batch):
        """
        Update the running window used to estimate marginals with the data in X
        """
        # Initialization
        if np.isnan(self.window[0, 0] ):
            # TO DO: avoid empty slice
            # Continuous columns: normal initialization
            if np.sum(self.cont_indices)>0:
                mean_cont = np.nanmean(X_batch[:, self.cont_indices])
                std_cont = np.nanstd(X_batch[:, self.cont_indices])
                if np.isnan(mean_cont):
                    self.window[:, self.cont_indices] = np.random.normal(0, 1, size=(self.window_size, np.sum(self.cont_indices)))
                else:
                    self.window[:, self.cont_indices] = np.random.normal(mean_cont, std_cont, size=(self.window_size, np.sum(self.cont_indices)))
            # Ordinal columns: uniform initialization
            for j,loc in enumerate(self.ord_indices):
                if loc:
                    min_ord = np.nanmin(X_batch[:, j])
                    max_ord = np.nanmax(X_batch[:,j]) 
                    if np.isnan(min_ord):
                        self.window[:,j].fill(0)
                    else:
                        self.window[:, j] = np.random.randint(min_ord, max_ord+1, size=self.window_size)
        # update for new data
        for row in X_batch:
            for col_num in range(len(row)):
                data = row[col_num]
                if not np.isnan(data):
                    self.window[self.update_pos[col_num], col_num] = data
                    self.update_pos[col_num] += 1 
                    if self.update_pos[col_num] >= self.window_size:
                        self.update_pos[col_num] = 0


    def partial_evaluate_cont_latent(self, X_batch):
        """
        Obtain the latent continuous values corresponding to X_batch 
        """
        X_cont = X_batch[:,self.cont_indices]
        window_cont = self.window[:,self.cont_indices]
        Z_cont = np.empty(X_cont.shape)
        Z_cont[:] = np.nan
        for i in range(np.sum(self.cont_indices)):
            # INPUT THE WINDOW FOR EVERY COLUMN
            missing = np.isnan(X_cont[:,i])
            Z_cont[~missing,i] = self.get_cont_latent(X_cont[~missing,i], window_cont[:,i])
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
        for i in range(np.sum(self.ord_indices)):
            missing = np.isnan(X_ord[:,i])
            # INPUT THE WINDOW FOR EVERY COLUMN
            Z_ord_lower[~missing,i], Z_ord_upper[~missing,i] = self.get_ord_latent(X_ord[~missing,i], window_ord[:,i])
        return Z_ord_lower, Z_ord_upper

    def partial_evaluate_cont_observed(self, Z_batch, X_batch=None):
        """
        Transform the latent continous variables in Z_batch into corresponding observations
        """
        Z_cont = Z_batch[:,self.cont_indices]
        if X_batch is None:
            X_batch = np.zeros(Z_batch.shape) * np.nan
        X_cont = X_batch[:,self.cont_indices]
        X_cont_imp = np.copy(X_cont)
        window_cont = self.window[:,self.cont_indices]
        for i in range(np.sum(self.cont_indices)):
            # if X_batch is not provided, missing will be 1:n
            missing = np.isnan(X_cont[:,i])
            if np.sum(missing)>0:
                X_cont_imp[missing,i] = self.get_cont_observed(Z_cont[missing,i], window_cont[:,i])
        return X_cont_imp

    def partial_evaluate_ord_observed(self, Z_batch, X_batch=None):
        """
        Transform the latent ordinal variables in Z_batch into corresponding observations
        """
        Z_ord = Z_batch[:,self.ord_indices]
        if X_batch is None:
            X_batch = np.zeros(Z_batch.shape) * np.nan
        X_ord = X_batch[:, self.ord_indices]
        X_ord_imp = np.copy(X_ord)
        window_ord = self.window[:,self.ord_indices]
        for i in range(np.sum(self.ord_indices)):
            missing = np.isnan(X_ord[:,i])
            if np.sum(missing)>0:
                X_ord_imp[missing,i] = self.get_ord_observed(Z_ord[missing,i], window_ord[:,i])
        return X_ord_imp

    def get_cont_latent(self, x_batch_obs, window):
        """
        Return the latent variables corresponding to the continuous entries of 
        self.X. Estimates the CDF columnwise with the empyrical CDF
        """
        ecdf = ECDF(window)
        l = len(window)
        q = (l / (l + 1.0)) * ecdf(x_batch_obs)
        q[q==0] = l/(l+1)/2
        if any(q==0):
            print("In get_cont_latent, 0 quantile appears")
        return norm.ppf(q)

    def get_cont_observed(self, z_batch_missing, window):
        """
        Applies marginal scaling to convert the latent entries in Z corresponding
        to continuous entries to the corresponding imputed oberserved value
        """
        quantiles = norm.cdf(z_batch_missing)
        return np.quantile(window, quantiles)

    def get_ord_latent(self, x_batch_obs, window):
        """
        get the cdf at each point in X_batch
        """
        # the lower endpoint of the interval for the cdf
        ecdf = ECDF(window)
        unique = np.unique(window)
        if unique.shape[0] > 1:
            threshold = np.min(np.abs(unique[1:] - unique[:-1]))/2.0
            z_lower_obs = norm.ppf(ecdf(x_batch_obs - threshold))
            z_upper_obs = norm.ppf(ecdf(x_batch_obs + threshold))
        else:
            z_upper_obs = np.inf
            z_lower_obs = -np.inf
            # If the window at j-th column only has one unique value, 
            # the final imputation will be the unqiue value regardless of the EM iteration.
            # In offline setting, we don't allow this happen.
            # In online setting, when it happens, 
            # we use -inf to inf to ensure tha EM iteration does not break down due to singularity
            print("window contains a single value")
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
