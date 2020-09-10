import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
class TransformFunction():
    def __init__(self, X, cont_indices, ord_indices):
        self.X = X
        self.ord_indices = ord_indices
        self.cont_indices = cont_indices

    def get_cont_latent(self):
        """
        Return the latent variables corresponding to the continuous entries of 
        self.X. Estimates the CDF columnwise with the empyrical CDF
        """
        X_cont = self.X[:,self.cont_indices]
        Z_cont = np.empty(X_cont.shape)
        n = self.X.shape[0]
        for i, x_col in enumerate(X_cont.T):
            missing = np.isnan(x_col)
            x_col_noNan = x_col[~missing]
            ecdf = ECDF(x_col_noNan)
            Z_cont[:,i] = norm.ppf((n / (n + 1.0)) * ecdf(x_col))
            # re-add the nan values
            Z_cont[missing,i] = np.nan
        return Z_cont

    def get_ord_latent(self):
        """
        Return the lower and upper ranges of the latent variables corresponding 
        to the ordinal entries of X. Estimates the CDF columnwise with the empyrical CDF
        """
        X_ord = self.X[:,self.ord_indices]
        Z_ord_lower = np.empty(X_ord.shape)
        Z_ord_upper = np.empty(X_ord.shape)
        for i, x_col in enumerate(X_ord.T):
            missing = np.isnan(x_col)
            x_col_noNan = x_col[~missing]
            ecdf = ECDF(x_col_noNan)
            unique = np.unique(x_col_noNan)
            # half the min differenence between two ordinals
            threshold = np.min(np.abs(unique[1:] - unique[:-1]))/2.0
            Z_ord_lower[:,i] = norm.ppf(ecdf(x_col - threshold))
            Z_ord_upper[:,i] = norm.ppf(ecdf(x_col + threshold))
            # re-add the nan values
            Z_ord_lower[missing,i] = np.nan
            Z_ord_upper[missing,i] = np.nan
        return Z_ord_lower, Z_ord_upper

    def impute_cont_observed(self, Z):
        """
        Applies marginal scaling to convert the latent entries in Z corresponding
        to continuous entries to the corresponding imputed oberserved value
        """
        X_cont = self.X[:, self.cont_indices]
        Z_cont = Z[:, self.cont_indices]
        X_imp = np.copy(X_cont)
        for i, x_col in enumerate(X_cont.T):
            missing = np.isnan(x_col)
            # Only impute missing entries
            X_imp[missing,i] = np.quantile(x_col[~missing], norm.cdf(Z_cont[missing,i]))
        return X_imp

    def impute_ord_observed(self, Z):
        """
        Applies marginal scaling to convert the latent entries in Z corresponding
        to ordinal entries to the corresponding imputed oberserved value
        """
        X_ord = self.X[:, self.ord_indices]
        Z_ord = Z[:, self.ord_indices]
        X_imp = np.copy(X_ord)
        for i, x_col in enumerate(X_ord.T):
            missing = np.isnan(x_col)
            # only impute missing entries
            X_imp[missing,i] = self.inverse_ecdf(x_col[~missing], norm.cdf(Z_ord[missing,i]))
        return X_imp

    def inverse_ecdf(self, data, x, DECIMAL_PRECISION = 3):
        """
        computes the inverse ecdf (quantile) for x with ecdf given by data
        """
        n = len(data)
        # round to avoid numerical errors in ceiling function
        quantile_indices = np.ceil(np.round_((n + 1) * x - 1, DECIMAL_PRECISION))
        quantile_indices = np.clip(quantile_indices, a_min=0,a_max=n-1).astype(int)
        sort = np.sort(data)
        return sort[quantile_indices]

