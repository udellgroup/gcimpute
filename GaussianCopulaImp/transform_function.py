import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, poisson
class TransformFunction():
    def __init__(self, X, cont_indices, ord_indices, poisson_cdf = None, poisson_inverse_cdf = None):
        '''
        This class performs transformation between the observed space and the latent space.
        '''
        self.X = X
        self.ord_indices = ord_indices
        self.cont_indices = cont_indices
        p = self.X.shape[1]
        self.cdf_type = np.array(['empirical'] * p)
        self.inverse_cdf_type = np.array(['empirical'] * p)
        if poisson_cdf is not None:
            try:
               self.cdf_type[poisson_cdf] = 'Poisson'
           except IndexError:
                print('Invalid poisson_cdf list value')
                raise
        if poisson_inverse_cdf is not None:
            try:
               self.inverse_cdf_type[poisson_inverse_cdf] = 'Poisson'
           except IndexError:
                print('Invalid poisson_inverse_cdf list value')
                raise


    def get_cont_latent(self, X_to_transform=None):
        """
        Return the latent variables corresponding to the continuous columns in X_to_transform.
        The marginal estimation is done using observation in self.X
        """
        indices = self.cont_indices
        X_to_est = self.X[:,indices]

        if X_to_transform is None:
            X_to_transform = X_to_est.copy()
        else:
            X_to_transform = X_to_transform[:,indices].copy()

        Z_cont = np.empty(X_to_transform.shape)
        for j, (x_col, x_est_col, cdf_type) in enumerate(zip(X_to_transform.T, X_to_est.T, self.cdf_type[indices])):
            Z_cont[:,j] = self._obs_to_latent_cont(x_to_transform = x_col, x_obs = x_est_col, cdf_type=cdf_type)
        return Z_cont

    def get_ord_latent(self, X_to_transform=None):
        """
        Return the latent variables corresponding to the ordinal columns in X_to_transform.
        The marginal estimation is done using observation in self.X.
        """
        indices = self.ord_indices
        X_to_est = self.X[:,indices]

        if X_to_transform is None:
            X_to_transform = X_to_est.copy()
        else:
            X_to_transform = X_to_transform[:,indices].copy()
        
        Z_ord_lower = np.empty(X_to_transform.shape)
        Z_ord_upper = np.empty(X_to_transform.shape)
        for j, (x_col, x_est_col, cdf_type) in enumerate(zip(X_to_transform.T, X_to_est.T, self.cdf_type[indices])):
            lower, upper = self._obs_to_latent_ord(x_to_transform = x_col, x_obs = x_est_col, cdf_type=cdf_type)
            Z_ord_lower[:,j] = lower
            Z_ord_upper[:,j] = upper
        return Z_ord_lower, Z_ord_upper

    def impute_cont_observed(self, Z, X_to_impute=None):
        """
        Return the observed variables corresponding to the continuous columns in Z,
        but only at the non-nan entries of X_to_impute.
        To return a complete matrix, input an all nan X_to_impute.
        The marginal estimation is done using observation in self.X.
        """
        indices = self.cont_indices
        if X_to_impute is None:
            X_to_impute = self.X[:, indices]
        else:
            X_to_impute = X_to_impute[:, indices]
        X_to_est = self.X[:, indices]
        Z_use = Z[:, indices]
        X_imp = X_to_impute.copy()
        for j, _tuple  in enumerate(zip(Z_use.T, X_to_est.T, X_to_impute.T, self.inverse_cdf_type[indices])):
            (z_col, x_est_col, x_imp_col, inverse_cdf_type) = _tuple
            X_imp[:,j] = self._latent_to_obs_cont(x_obs = x_est_col, z_latent = z_col, x_to_impute = x_imp_col, inverse_cdf_type=inverse_cdf_type)
        return X_imp

    def impute_ord_observed(self, Z, X_to_impute=None):
         """
        Return the observed variables corresponding to the ordinal columns in Z,
        but only at the non-nan entries of X_to_impute.
        To return a complete matrix, input an all nan X_to_impute.
        The marginal estimation is done using observation in self.X.
        """
        indices = self.ord_indices
        if X_to_impute is None:
            X_to_impute = self.X[:, indices]
        else:
            X_to_impute = X_to_impute[:, indices]
        X_to_est = self.X[:, indices]
        Z_use = Z[:, indices]
        X_imp = X_to_impute.copy()
        for j, _tuple in enumerate(zip(Z_use.T, X_to_est.T, X_to_impute.T, self.inverse_cdf_type[indices])):
            (z_col, x_est_col, x_imp_col, inverse_cdf_type) = _tuple
            X_imp[:,j] = self._latent_to_obs_ord(x_obs = x_est_col, z_latent = z_col, x_to_impute = x_imp_col, inverse_cdf_type=inverse_cdf_type)
        return X_imp

    def _obs_to_latent_cont(self, x_to_transform, x_obs, cdf_type='empirical'):
        '''
        Transform observed to latent for a single variable
        '''
        out = np.empty_like(x_to_transform)
        missing = np.isnan(x_to_transform)
        # TODO: remember marginal estimation for each column
        f_marginal = self._marginal_cont_est(x_obs = x_obs, cdf_type=cdf_type)
        out[~missing] = f_marginal(x_to_transform[~missing])
        out[missing] = np.nan
        return out

    def _obs_to_latent_ord(self, x_to_transform, x_obs, cdf_type='empirical'):
        '''
        Transform observed to latent for a single variable
        '''
        lower = np.empty_like(x_to_transform)
        upper = np.empty_like(x_to_transform)
        missing = np.isnan(x_to_transform)
        # TODO: remember marginal estimation for each column
        f_marginal = self._marginal_ord_est(x_obs = x_obs, cdf_type=cdf_type)
        lower[~missing], upper[~missing] = f_marginal(x_to_transform[~missing])
        lower[missing], upper[missing] = np.nan, np.nan
        return lower, upper

    # TODO: merge _latent_to_obs_cont and _latent_to_obs_ord
    def _latent_to_obs_cont(self, x_obs, z_latent, x_to_impute=None, inverse_cdf_type='empirical'):
        '''
        Transform latent to observed for a single variable
        '''
        if x_to_impute is None:
            x_to_impute = x_obs
        x_imp = x_to_impute.copy()
        missing = np.isnan(x_to_impute)

        f_inverse_marginal = self._inverse_marginal_cont_est(x_obs = x_obs, inverse_cdf_type=inverse_cdf_type)
        # TODO: try different interpolation strategy
        if any(missing):
            x_imp[missing] = f_inverse_marginal(z_latent[missing])
        return x_imp

    def _latent_to_obs_ord(self, x_obs, z_latent, x_to_impute=None, inverse_cdf_type='empirical'):
        '''
        Transform latent to observed for a single variable

        Args:
            x_obs: (nsample, )
                Used for marginal estimate
            z_latent: (nsample, )
                Used for quantile numbers computation
            x_to_impute: (nsample, ) or None
                Used for detect entries to be computed
        '''
        if x_to_impute is None:
            x_to_impute = x_obs
        x_imp = x_to_impute.copy()
        missing = np.isnan(x_to_impute)

        f_inverse_marginal = self._inverse_marginal_ord_est(x_obs = x_obs, inverse_cdf_type=inverse_cdf_type)
        # TODO: try different interpolation strategy
        if any(missing):
            x_imp[missing] = f_inverse_marginal(z_latent[missing])
        return x_imp

    def _marginal_cont_est(self, x_obs, cdf_type='empirical'):
        x_obs = x_obs[~np.isnan(x_obs)]
        l = len(x_obs)
        assert l>0, 'Each variable must have at least one observation'
        if cdf_type == 'empirical':
            func = ECDF(x_obs)
        else:
            raise NotImplemented("Only 'empirical' and 'Poisson' are allowed for marginal CDF estimation")
        def marginal(x):
            # from obs to scores
            q = func(x)
            # avoid 0 and 1
            q *= l / (l+1)
            q[q==0] = 1/(2*(l+1))
            # to normal quantiles 
            q = norm.ppf(q)
            return q
        return marginal

    def _marginal_ord_est(self, x_obs, cdf_type='empirical'):
        x_obs = x_obs[~np.isnan(x_obs)]
        unique = np.unique(x_obs)
        l = len(x_obs)
        assert l>0, 'Each variable must have at least one observation'
        if cdf_type == 'empirical':
            func = ECDF(x_obs)
            threshold = np.min(np.abs(unique[1:] - unique[:-1]))/2.0
        elif cdf_type == 'Poisson':
            func = lambda x, mu=x_obs.mean(): poisson.cdf(x, mu = mu)
            # for count data, the corresponding latent interval will be shorter than ordinal treatment
            # refelecting stronger regularization due to parametric model
            threshold = 0.5
        else:
            raise NotImplemented("Only 'empirical' and 'Poisson' are allowed for marginal CDF estimation")
        def marginal(x):
            if len(unique)>1:
                # from obs to scores
                lower, upper = func(x-threshold), func(x+threshold)
                lower, upper = norm.ppf(lower), norm.ppf(upper)
            else:
                print("Some ordinal variable has only a single level")
                lower = np.ones_like(x) - np.inf
                upper = np.ones_like(x) + np.inf
            return lower, upper
        return marginal

    def _inverse_marginal_cont_est(self, x_obs, inverse_cdf_type='empirical'):
        '''
        Return the estimated inverse marginal from z to x for a continuous variable.
        '''
        x_obs = x_obs[~np.isnan(x_obs)]
        l = len(x_obs)
        assert l>0, 'Each variable must have at least one observation'

        if inverse_cdf_type == 'empirical':
            func = lambda q, x_obs=x_obs: np.quantile(x_obs, q)
        else:
            raise NotImplemented("Only 'empirical' is allowed for marginal inverse CDF estimation")

        def inverse_marginal(z):
            q = norm.cdf(z)
            x = func(q)
            return x

        return inverse_marginal


    def _inverse_marginal_ord_est(self, x_obs, inverse_cdf_type='empirical'):
        '''
        Return the estimated inverse marginal from z to x for an ordinal variable.
        '''
        x_obs = x_obs[~np.isnan(x_obs)]
        l = len(x_obs)
        assert l>0, 'Each variable must have at least one observation'

        if inverse_cdf_type == 'empirical':
            func = lambda q, x_obs=x_obs: self.inverse_ecdf(data=x_obs, x=q)
        elif inverse_cdf_type == 'Poisson':
            func = lambda q, mu=x_obs.mean(): poisson.ppf(q, mu=mu)
        else:
            raise NotImplemented("Only 'empirical' and 'Poisson' are allowed for marginal inverse CDF estimation")

        def inverse_marginal(z):
            q = norm.cdf(z)
            x = func(q)
            return x

        return inverse_marginal

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

