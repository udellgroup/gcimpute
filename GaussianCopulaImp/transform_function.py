import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
class TransformFunction():
    def __init__(self, X, cont_indices, ord_indices):
        self.X = X
        self.ord_indices = ord_indices
        self.cont_indices = cont_indices

    def get_cont_latent(self, X_to_transform=None):
        """
        Return the latent variables corresponding to the continuous entries of 
        self.X. Estimates the CDF columnwise with the empyrical CDF
        """
        indices = self.cont_indices
        X_to_est = self.X[:,indices]

        if X_to_transform is None:
            X_to_transform = X_to_est.copy()
        else:
            X_to_transform = X_to_transform[:,indices].copy()

        Z_cont = np.empty(X_to_transform.shape)
        for j, (x_col, x_est_col) in enumerate(zip(X_to_transform.T, X_to_est.T)):
            Z_cont[:,j] = self._obs_to_latent_cont(x_obs = x_col, x_ecdf = x_est_col)
        return Z_cont

    def get_ord_latent(self, X_to_transform=None):
        """
        Return the lower and upper ranges of the latent variables corresponding 
        to the ordinal entries of X. Estimates the CDF columnwise with the empyrical CDF
        """
        indices = self.ord_indices
        X_to_est = self.X[:,indices]

        if X_to_transform is None:
            X_to_transform = X_to_est.copy()
        else:
            X_to_transform = X_to_transform[:,indices].copy()
        
        Z_ord_lower = np.empty(X_to_transform.shape)
        Z_ord_upper = np.empty(X_to_transform.shape)
        for j, (x_col, x_est_col) in enumerate(zip(X_to_transform.T, X_to_est.T)):
            lower, upper = self._obs_to_latent_ord(x_obs = x_col, x_ecdf = x_est_col)
            Z_ord_lower[:,j] = lower
            Z_ord_upper[:,j] = upper
        return Z_ord_lower, Z_ord_upper

    def impute_cont_observed(self, Z, X_to_impute=None):
        """
        Applies marginal scaling to convert the latent entries in Z corresponding
        to continuous entries to the corresponding imputed oberserved value
        """
        indices = self.cont_indices
        if X_to_impute is None:
            X_to_impute = self.X[:, indices]
        else:
            X_to_impute = X_to_impute[:, indices]
        X_to_est = self.X[:, indices]
        Z_use = Z[:, indices]
        X_imp = X_to_impute.copy()
        for j, (z_col, x_est_col, x_imp_col)  in enumerate(zip(Z_use.T, X_to_est.T, X_to_impute.T)):
            X_imp[:,j] = self._latent_to_obs_cont(x_obs = x_est_col, z_latent = z_col, x_to_impute=x_imp_col)
        return X_imp

    def impute_ord_observed(self, Z, X_to_impute=None):
        """
        Applies marginal scaling to convert the latent entries in Z corresponding
        to ordinal entries to the corresponding imputed oberserved value.
        For each variable, the missing entries are detected by reading the storeed marginal estimate points
        """
        indices = self.ord_indices
        if X_to_impute is None:
            X_to_impute = self.X[:, indices]
        else:
            X_to_impute = X_to_impute[:, indices]
        X_to_est = self.X[:, indices]
        Z_use = Z[:, indices]
        X_imp = X_to_impute.copy()
        for j, (z_col, x_est_col, x_imp_col)  in enumerate(zip(Z_use.T, X_to_est.T, X_to_impute.T)):
            X_imp[:,j] = self._latent_to_obs_ord(x_obs = x_est_col, z_latent = z_col, x_to_impute=x_imp_col)
        return X_imp

    def _obs_to_latent_cont(self, x_obs, x_ecdf):
        x_ecdf = x_ecdf[~np.isnan(x_ecdf)]
        ecdf = ECDF(x_ecdf)
        l = len(x_ecdf)
        q = (l / (l + 1.0)) * ecdf(x_obs)
        q[q==0] = 1/(2*(l+1))
        q = norm.ppf(q)
        q[np.isnan(x_obs)] = np.nan
        return q

    def _obs_to_latent_ord(self, x_obs, x_ecdf):
        x_ecdf = x_ecdf[~np.isnan(x_ecdf)]
        ecdf = ECDF(x_ecdf)
        unique = np.unique(x_ecdf)
        # half the min differenence between two ordinals
        if len(unique)>1:
            threshold = np.min(np.abs(unique[1:] - unique[:-1]))/2.0
            lower = norm.ppf(ecdf(x_obs - threshold))
            upper = norm.ppf(ecdf(x_obs + threshold))
        else:
            print("Only a single level for ordinal variable")
            lower = np.ones_like(x_obs) - np.inf
            upper = np.ones_like(x_obs) + np.inf
        missing = np.isnan(x_obs)
        lower[missing] = np.nan
        upper[missing] = np.nan
        return lower, upper

    # TODO: merge _latent_to_obs_cont and _latent_to_obs_ord
    def _latent_to_obs_cont(self, x_obs, z_latent, x_to_impute=None):
        if x_to_impute is None:
            x_imp = x_obs.copy()
            missing = np.isnan(x_obs)
            x_obs_est = x_obs[~missing]
        else:
            x_imp = x_to_impute.copy()
            missing = np.isnan(x_to_impute)
            x_obs_est= x_obs[~np.isnan(x_obs)]
        # TODO: try different interpolation strategy
        # the imputation function
        # only impute missing entries
        if any(missing):
            q_imp = norm.cdf(z_latent[missing])
            x_imp[missing] = np.quantile(x_obs_est, q_imp)
        return x_imp

    def _latent_to_obs_ord(self, x_obs, z_latent, x_to_impute=None):
        '''
        Transform latent values into the observed space for a variable.
        Args:
            x_obs: (nsample, )
                Used for marginal estimate
            z_latent: (nsample, )
                Used for quantile numbers computation
            x_to_impute: (nsample, ) or None
                Used for detect entries to be computed
        '''
        if x_to_impute is None:
            x_imp = x_obs.copy()
            missing = np.isnan(x_obs)
            x_obs_est = x_obs[~missing]
        else:
            x_imp = x_to_impute.copy()
            missing = np.isnan(x_to_impute)
            x_obs_est = x_obs[~np.isnan(x_obs)]
        # TODO: try different interpolation strategy
        # the imputation function
        # only impute missing entries
        if any(missing):
            q_imp = norm.cdf(z_latent[missing])
            x_imp[missing] = self.inverse_ecdf(data=x_obs_est, x=q_imp)
        return x_imp

    def inverse_ecdf(self, data, x, DECIMAL_PRECISION = 3):
        """
        computes the inverse ecdf (quantile) for x with ecdf given by data
        """
        n = len(data)
        if n==0:
            print('No observation can be used for imputation')
            raise
        # round to avoid numerical errors in ceiling function
        quantile_indices = np.ceil(np.round_((n + 1) * x - 1, DECIMAL_PRECISION))
        quantile_indices = np.clip(quantile_indices, a_min=0,a_max=n-1).astype(int)
        sort = np.sort(data)
        return sort[quantile_indices]

