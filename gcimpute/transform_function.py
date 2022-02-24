import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, poisson
from functools import partial
from .marginal_imputation import *
from bisect import bisect
#import warnings
#warnings.filterwarnings("error")

class TransformFunction():
    '''
    Transformation operation.
    This class performs transformation between the observed space and the latent space.

    Parameters
    ----------
    X: array-like of shape (nsamples, nfeatures)
        Input data, used to estimate the desired transformation
    cont_indices: array-like of shape (nfeatures,)
        Each entry is bool: indicating whether the corresponding variable is continuous
    ord_indices: array-like of shape (nfeatures,)
        Must be ~cont_indices
    cdf_types: array-like of shape (nfeatures,)
        Each entry is str in {'empirical', 'poisson', 'lower_truncated', 'upper_truncated', 'twosided_truncated'}.
        Indicate the estimation strategy for the empirical CDF
    inverse_cdf_types: array-like of shape (nfeatures,)
        Each entry is str in {'empirical', 'poisson', 'lower_truncated', 'upper_truncated', 'twosided_truncated'}.
        Indicate the estimation strategy for the empirical quantile
    '''
    
    def __init__(self, X, cont_indices, ord_indices, cdf_types, inverse_cdf_types):
        self.X = X
        self.ord_indices = ord_indices
        self.cont_indices = cont_indices
        p = self.X.shape[1]
        self.cdf_type = cdf_types
        self.inverse_cdf_type = inverse_cdf_types

        self.decay_weights = None

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
        Compute the observed variables corresponding to the continuous columns in Z,
        but only at the non-nan entries of X_to_impute.
        The marginal estimation is done using observation in self.X.
        To return a complete matrix, input an all np.nan X_to_impute.

        Parameters
        ----------
            Z: array-like of shape (nsamples, nfeatures)
                complete matrix in latent Gaussian space
            X_to_impute: array-like of shape (nsamples, nfeatures) or None
                Used to indicate the locations to be kept. 
                If None, set to the contniuous columns of the stored data
        
        Returns
        -------
            X_imp: array-like of shape (nsamples, nfeatures)
                The completed matrix in the observed space.
                At the observed entries of X_to_impute, X_imp agrees with X_to_impute.
                At the missing entries of X_to_impute, X_imp is transformed from Z

        """
        indices = self.cont_indices
        if X_to_impute is None:
            X_to_impute = self.X.copy()
        X_to_impute = X_to_impute[:, indices]
        X_to_est = self.X[:, indices]
        Z_use = Z[:, indices]
        X_imp = X_to_impute.copy()
        for j, _tuple  in enumerate(zip(Z_use.T, X_to_est.T, X_to_impute.T, self.inverse_cdf_type[indices])):
            (z_col, x_est_col, x_imp_col, inverse_cdf_type) = _tuple
            X_imp[:,j] = self._latent_to_obs_cont(x_obs = x_est_col, z_latent = z_col, x_to_impute = x_imp_col, 
                                                  inverse_cdf_type=inverse_cdf_type, 
                                                  weights = self.decay_weights)
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
        f_marginal = self._marginal_ord_est(x_obs = x_obs, cdf_type=cdf_type)
        l, u = f_marginal(x_to_transform[~missing])

        if len(l)>0 and (u-l).min()<=0:
            print('Invalid lower & upper bounds for ordinal')
            loc = np.argmin(u-l)
            print(f'Min of upper - lower: {u[loc]-l[loc]:.3f}')
            print(f'where upper is {u[loc]:.3f} and lower is {l[loc]:.3f}')
            print('The empirical unique observations: ')
            print(np.unique(x_obs))
            print('To be transformed unique values: ')
            print(np.unique(x_to_transform[~missing]))
            raise ValueError

        lower[~missing], upper[~missing] = l, u
        lower[missing], upper[missing] = np.nan, np.nan
        return lower, upper

    # TODO: merge _latent_to_obs_cont and _latent_to_obs_ord
    def _latent_to_obs_cont(self, x_obs, z_latent, x_to_impute=None, inverse_cdf_type='empirical', weights = None):
        '''
        Transform latent to observed for a single variable

        Parameters
        ----------
            x_obs: array-like of shape (nsamples, )
                The stored observed variable, used for marginal estimation
            z_latent: array-like of shape (nsamples, )
                The latent vector to be transformed into the observed space 
            x_to_impute: array-like of shape (nsamples, ) or None 
                Used to indicate the locations to be kept. 
                z_latent will only be transformed into the observed space at the missing entries of x_to_impute
                If none, set to x_obs. 

        Returns
        -------
            x_imp: array-like of shape (nsamples, )
                The completed vector.
                At the observed entries of x_to_impute, x_imp agrees with x_to_impute.
                At the missing entries of x_to_impute, x_imp is transformed from z_latent

        """
        '''
        if x_to_impute is None:
            x_to_impute = x_obs
        x_imp = x_to_impute.copy()
        missing = np.isnan(x_to_impute)

        f_inverse_marginal = self._inverse_marginal_cont_est(x_obs = x_obs, inverse_cdf_type=inverse_cdf_type, weights = weights)
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
            raise NotImplemented("Only 'empirical' and 'poisson' are allowed for marginal CDF estimation")
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
        unique = np.sort(np.unique(x_obs))
        _max, _min = unique[-1], unique[0]
        l = len(unique)
        assert l>1, 'Each ordinal variable must have at least two unique observations'
        assert _max > _min
        if cdf_type == 'empirical':
            func = ECDF(x_obs)
            threshold = np.min(np.abs(unique[1:] - unique[:-1]))/2.0
        elif cdf_type == 'poisson':
            func = lambda x, mu=x_obs.mean(): poisson.cdf(x, mu = mu)
            # for count data, the corresponding latent interval will be shorter than ordinal treatment
            # refelecting stronger regularization due to parametric model
            threshold = 0.5
        elif cdf_type not in ['lower_truncated','upper_truncated','twosided_truncated']:
            print(f"Invalid ordinal marginal estimation argument: {cdf_type}")
            raise NotImplementedError

        def marginal(x):
            # it may happen that some test points (x_to_transform) in ordinal variables do not appear in training points (x_obs)
            x[x<_min] = _min
            x[x>_max] = _max
            # from obs to scores
            lower, upper = func(x-threshold), func(x+threshold)
            # on unobserved index
            index = np.flatnonzero(lower == upper)
            for i in index:
                _i = bisect(unique, x[i])
                if _i==0 or _i>=l:
                    print(f'unique values: {unique}')
                    print(f'unobserved value: {x[i]}')
                    print(f'bisect index: {_i}')
                    raise ValueError
                lower[i] = func(unique[_i-1])
                upper[i] = func(unique[_i])

            # 
            lower, upper = norm.ppf(lower), norm.ppf(upper)
            return lower, upper

        if cdf_type in ['empirical','poisson']:
            f_marginal = marginal
        elif cdf_type == 'lower_truncated':
            f_marginal = partial(truncated_marginal_lower, x_obs = x_obs)
        elif cdf_type == 'upper_truncated':
            f_marginal = partial(truncated_marginal_upper, x_obs = x_obs)
        elif cdf_type == 'twosided_truncated':
            f_marginal = partial(truncated_marginal_twoside, x_obs = x_obs)

        return f_marginal

    def _inverse_marginal_cont_est(self, x_obs, inverse_cdf_type='empirical', weights=None):
        '''
        Return the estimated inverse marginal from z to x for a continuous variable.
        '''
        x_obs = x_obs[~np.isnan(x_obs)]
        l = len(x_obs)
        assert l>0, 'Each variable must have at least one observation'

        if inverse_cdf_type == 'empirical':
            func = self._empirical_cont_quantile(x_obs, weights = weights)
        else:
            raise NotImplemented("Only 'empirical' is allowed for marginal inverse CDF estimation")

        def inverse_marginal(z):
            q = norm.cdf(z)
            x = func(q)
            return x

        return inverse_marginal

    def _empirical_cont_quantile(self, x_obs, weights=None):
        if weights is None:
            func = lambda q, x_obs=x_obs: np.quantile(x_obs, q)
        else:
            assert len(weights) == len(x_obs), 'inconsistent sample weights'
            func = lambda q, x_obs=x_obs: weighted_quantile(values = x_obs, quantiles = q, sample_weight=weights)
        return func


    def _inverse_marginal_ord_est(self, x_obs, inverse_cdf_type='empirical'):
        '''
        Return the estimated inverse marginal from z to x for an ordinal variable.
        '''
        x_obs = x_obs[~np.isnan(x_obs)]
        l = len(x_obs)
        assert l>0, 'Each variable must have at least one observation'

        if inverse_cdf_type == 'empirical':
            func = partial(inverse_ecdf, x_obs=x_obs)
        elif inverse_cdf_type == 'poisson':
            func = lambda q, mu=x_obs.mean(): poisson.ppf(q, mu=mu)
        elif inverse_cdf_type == 'lower_truncated':
            func = partial(truncated_inverse_marginal_lower, x_obs=x_obs)
        elif inverse_cdf_type == 'upper_truncated':
            func = partial(truncated_inverse_marginal_upper, x_obs=x_obs)
        elif inverse_cdf_type == 'twosided_truncated':
            func = partial(truncated_inverse_marginal_twoside, x_obs=x_obs)
        else:
            raise NotImplemented("Invalid ordinal inverse marginal estimation argument")

        def inverse_marginal(z):
            q = norm.cdf(z)
            x = func(q)
            return x

        return inverse_marginal

    


