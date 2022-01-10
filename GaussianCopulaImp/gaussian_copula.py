from .transform_function import TransformFunction
from .online_transform_function import OnlineTransformFunction
from .embody import _latent_operation_body_
from scipy.stats import norm, truncnorm
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.linalg import svdvals
from collections import defaultdict
import warnings

var_type_names = ['continuous', 'ordinal', 'lower_truncated', 'upper_truncated', 'twosided_truncated']

class GaussianCopula():
    '''
    Gaussian copula model.
    This class allows to estimate the parameters of a Gaussian copula model from incomplete data, 
    and impute the missing entries using the learned model.


    Attributes
    ----------
    cont_indices: ndarray 
        Set of continuous variable indices.
    ord_indices: ndarray
        Set of ordinal variable indices. The complement set of cont_indices.
    n_iter_: int
        Number of iteration rounds that occurred. Will be less than self._max_iter if early stopping criterion was reached.
    likelihood: list of length n_iter_
        The computed pseudo likelihood value at each iteration.
    feature_names: ndarray of shape n_features
        Names of features seen during fit. Defined only when X has feature names that are all strings.
    corr_diff: list of length 0 or n_iter_
        The changing tracking statistics of the copula correlation matrix if training_mode is 'minibatch-online',
        and an empty list otherwise.

    Methods
    -------
    fit(X)
        Fit a Gaussian copula model on X.
    transform(X)
        Return the imputed X using the stored model.
    fit_transform(X)
        Fit a Gaussian copula model on X and return the transformed X.
    sample_imputation(X)
        Return multiple imputed datasets X using the stored model.
    get_params()
        Get parameters for this estimator.
    get_imputed_confidence_interval(alpha=0.95)
        Get the confidence intervals for the imputed missing entries.
    get_reliability(Ximp=None, alpha=0.95)
        Get the reliability, a relative quantity across all imputed entries, when either all variables are continuous or all variables are ordinal 

    '''

    def __init__(self, training_mode='standard', stepsize_func=lambda k, c=5:c/(k+c), const_stepsize=0.5, batch_size=100, window_size=200, decay=None, realtime_marginal=True, min_ord_ratio=0.1, tol=0.01, likelihood_min_increase=0, max_iter=50, num_pass=2, random_state=101, n_jobs=1, verbose=0, num_ord_updates=1, corr_diff_type=['F'], use_truncation_var=True):
        '''
        Parameters:
            training_mode: {'standard', 'minibatch-offline', 'minibatch-online'}, default='standard'
                String describing the type of training to use. Must be one of:
                'standard'
                    all data are used to estimate the marginals and update the model in each iteration
                'minibatch-offline'
                    all data are used to estimate the marginals, but only a mini-batch's data are used to update the model in each iteration
                'minibatch-online'
                    only recent data are used to estimate the marginals, and only a mini-batch's data are used to update the model in each iteration
            stepsize_func: a function that outputs monotonically decreasing values in the range (0,1) on positive integers
                Only used when (1) training_mode = 'minibatch-offline'; (2) training_mode = 'minibatch-online' and 'const_stepsize=None'.
            const_stepsize: float in the range (0,1) or None, default is 0.5.
                Only used when training_mode = 'minibatch-online'. 
            batch_size: int, default=100
                The number of data points in each mini-batch. Only used for offline mini-batch training.
            window_size: int, default=200
                The lookback window length for online marginal estimate. Only used when training_mode = 'minibatch-online'.  
            min_ord_ratio: float, default=0.1
                When cont_indices is None when the calling fit, variables whose mode occurence ratio is smaller than min_ord_ratio are regarded as continuous variables.
            tol: float, default=0.001
                The convergence threshold. EM iterations will stop when the parameter update ratio is below this threshold.
            likelihood_min_increase: float, default=0.01
                The minimal likelihood increase ratio required to keep running the EM algorithm.
            max_iter: int, default=30
                The number of EM iterations to perform.
            num_pass: int or None, default=2
                Only used when training_mode='minibatch-offline'. Used to set max_iter.
            random_state: int, default=101
                Controls the randomness in generating latent ordinal values. Not used if there is no ordinal variable.
            n_jobs: int, default=1
                The number of jobs to run in parallel.
            verbose: int, default=0
                Controls the verbosity when fitting and predicting. 
            num_ord_updates: int, default=1
                Number of steps to take when approximating the mean and variance of the latent variables corresponding to ordinal dimensions.
                We do not recommend using value larger than 1 (the default value) at this moment. It will slow the speed without clear 
                performance improvement.
            corr_diff_type: A list with elements from {'F', 'S', 'N'}, default = ['F']
                The matrix norm used to compute copula correlation update ratio. Used for detecting change points when training mode = 'minibatch-online'. 
                Must be one of:
                'F'
                    Frobenius norm
                'S'
                    Spectral norm
                'N'
                    Nuclear norm
        '''
        def check_stepsize():
            L = np.array([stepsize_func(x) for x in range(1, max_iter+1, 1)])
            if L.min() <=0 or L.max()>=1:
                print(f'Step size should be in the range of (0,1). The input stepsize function yields step size from {L.min()} to {L.max()}')
                raise
            if not all(x>y for x, y in zip(L, L[1:])):
                print(f'Input step size is not monotonically decreasing.')
                raise
        
        if training_mode == 'minibatch-online':
            if const_stepsize is None:
                check_stepsize()
                self.stepsize = stepsize_func
            else:
                assert 0<const_stepsize<1, 'const_stepsize must be in the range (0, 1)'
                self.stepsize = lambda x, c=const_stepsize: c
        elif training_mode == 'minibatch-offline':
            check_stepsize()
            self.stepsize = stepsize_func
        elif training_mode == 'standard':
            pass
        else:
            print("Invalida training_mode, must be one of 'standard', 'minibatch-offline', 'minibatch-online'")
            raise

        self._training_mode = training_mode
        self._batch_size = batch_size
        self._window_size = window_size
        self._realtime_marginal = realtime_marginal
        self._decay = decay
        self._corr_diff_type = corr_diff_type

        # self._cont_indices and self._ord_indices store boolean indexing
        # self.cont_indices and self.ord_indices store integer indexing
        self._cont_indices = None
        self._ord_indices = None 
        self.cont_indices = None
        self.ord_indices = None 
        self._min_ord_ratio = min_ord_ratio
        self.var_type_dict = {}
        self.use_truncation_var = use_truncation_var

        self._seed = random_state
        self._rng = np.random.default_rng(self._seed)
        self._sample_seed = self._seed

        self._threshold = tol
        self._likel_threshold = likelihood_min_increase
        self._max_iter = max_iter
        self._max_workers = n_jobs
        self._verbose = verbose
        self._num_ord_updates = num_ord_updates
        self._num_pass = num_pass

        self._iter = 0

        # model parameter
        self._corr = None
        
        # attributes
        self.n_iter_ = 0
        self.likelihood = []
        self.corrupdate = []
        self.features_names = None
        self.corr_diff = defaultdict(list)

    ################################################
    #### public functions
    ################################################

    def fit(self, X, 
            continuous = None,
            ordinal = None, 
            lower_truncated= None, 
            upper_truncated = None, 
            twosided_truncated = None,
            **kwargs):
        '''
        Fits the Gaussian copula imputer on the input data X.

        Parameters:
            X: array-like of shape (n_samples, n_features)
                Input data
            ordinal, lower_truncated, upper_truncated, twosided_truncated, poisson: list of the corresponding variable type indices
            kwargs:
                additional keyword arguments for fit_offline
        '''
        self.store_var_type(continuous = continuous,
                            ordinal = ordinal, 
                            lower_truncated = lower_truncated,
                            upper_truncated = upper_truncated,
                            twosided_truncated = twosided_truncated
                           )

        if self._training_mode == 'minibatch-online':
            print('fit method is not implemented for minibatch-online mode, since the fitting and imputation are done in the unit of mini-batch. To impute the missing entries, call fit_transform.')
            raise
        else:
            
            return self.fit_offline(X, **kwargs)

    def transform(self, X=None, num_ord_updates=2):
        '''
        Impute the missing entries in X using currently fitted model (accessed through self._corr). 
        If X is None, set X as the data used to fit the model.
        '''
        # get Z
        if X is None:
            Z = self._latent_Zimp
        else:
            Z, Z_ord_lower, Z_ord_upper = self._observed_to_latent(X_to_transform=X)
            Z, _ = self._fillup_latent(Z=Z, Z_ord_lower=Z_ord_lower, Z_ord_upper=Z_ord_upper, num_ord_updates=num_ord_updates)
        # from Z to X
        X_imp = self._latent_to_imp(Z=Z, X_to_impute=X)
        return X_imp

    def fit_transform(self, X, 
                      continuous = None,
                      ordinal = None, 
                      lower_truncated= None, 
                      upper_truncated = None, 
                      twosided_truncated = None,
                      **kwargs
                     ):
        '''
        Fit to data, then transform it. 
        For 'minibatch-online' mode, the variable types are set in this function call since the fit and transformation are done in an alternative fashion.
        For the other two modes, the variable types are set in the function fit_offline.

        Parameters:
            X: array-like of shape (n_samples, n_features)
                Input data
            cont_indices: array-list of int, optional (default=None)
                If not None, the set of continuout variable indices. 
                If None, the continuout variable indices will be determined by self.min_ord_ratio.
        Returns:
            Ximp: array-like of shape (n_samples, n_features)
                The imputed input data
        '''
        self.store_var_type(continuous = continuous, 
                            ordinal = ordinal, 
                            lower_truncated = lower_truncated,
                            upper_truncated = upper_truncated,
                            twosided_truncated = twosided_truncated
                           )
        

        if self._training_mode == 'minibatch-online':
            X = self._preprocess_data(X, set_indices=False)
            if 'X_true' in kwargs:
                self.set_indices(np.asarray(kwargs['X_true']))
            else:
                self.set_indices(X)
            kwargs_online = {name:kwargs[name] for name in ['n_train', 'X_true'] if name in kwargs}
            return self.fit_transform_online(X, **kwargs_online)
        else:
            X = self._preprocess_data(X)
            kwargs_offline = {name:kwargs[name] for name in ['first_fit', 'max_iter', 'convergence_verbose'] if name in kwargs}
            return self.fit_transform_offline(X, **kwargs_offline)

    def fit_transform_evaluate(self, X, eval_func=None, num_iter=30, return_Ximp=False, **kwargs):
        '''
        Run the algorithm for num_iter iterations and evaluate the returned imputed sample at each iteration.
        '''
        out = defaultdict(list)
        # first fit
        Ximp = self.fit_transform(X = X, max_iter = 1, convergence_verbose = False, **kwargs)
        if eval_func is not None:
            out['evaluation'].append(eval_func(Ximp))
        if return_Ximp:
            out['X_imp'].append(Ximp)

        # subsequent fits
        for i in range(1, num_iter, 1):
            Ximp = self.fit_transform(X = X, max_iter = 1, first_fit = False, convergence_verbose = False)
            if eval_func is not None:
                out['evaluation'].append(eval_func(Ximp))
            if return_Ximp:
                out['X_imp'].append(Ximp)

        return out

    def get_params(self):
        '''
        Get parameters for this estimator.

        Returns:
            params: dict
        '''
        params = {'copula_corr': self._corr.copy()}
        return params

    def get_imputed_confidence_interval(self, X=None, alpha = 0.95, num_ord_updates=1, type='analytical', num=200):
        '''
        Compute the confidence interval for each imputed entry.
        '''
        if self._training_mode == 'minibatch-online':
            raise NotImplementedError('Confidence interval has not yet been supported for minibatch-online mode')

        if type == 'quantile':
            return self.get_imputed_confidence_interval_quantile(X=X, alpha=alpha, num_ord_updates=num_ord_updates, num=num)

        if X is None:
            Zimp = self._latent_Zimp
            Cord = self._latent_Cord
            X = self.transform_function.X
        else:
            Z, Z_ord_lower, Z_ord_upper = self._observed_to_latent(X_to_transform=X)
            Zimp, Cord = self._fillup_latent(Z=Z, Z_ord_lower=Z_ord_lower, Z_ord_upper=Z_ord_upper, num_ord_updates=num_ord_updates)
            
        n, p = Zimp.shape
        margin = norm.ppf(1-(1-alpha)/2)

        # upper and lower have np.nan at oberved locations because std_cond has np.nan at those locations
        std_cond = self._get_cond_std_missing(X=X, Cord=Cord)
        upper = Zimp + margin * std_cond
        lower = Zimp - margin * std_cond

        # monotonic transformation
        upper = self._latent_to_imp(Z=upper, X_to_impute=X)
        lower = self._latent_to_imp(Z=lower, X_to_impute=X)
        obs_loc = ~np.isnan(X)
        upper[obs_loc] = np.nan
        lower[obs_loc] = np.nan
        return {'upper':upper, 'lower':lower}

    def get_imputed_confidence_interval_quantile(self, X=None, alpha = 0.95, num_ord_updates=1, num=200):
        '''
        Construct quantile based confidence interval
        '''
        if X is None:
            X =  self.transform_function.X
        X_imp_num = self.sample_imputation(X = X, num = num, num_ord_updates = num_ord_updates)
        q_lower, q_upper = (1-alpha)/2, 1-(1-alpha)/2
        lower, upper = np.quantile(X_imp_num, [q_lower, q_upper], axis=2)
        obs_loc = ~np.isnan(X)
        upper[obs_loc] = np.nan
        lower[obs_loc] = np.nan
        return {'upper':upper, 'lower':lower}


    def get_reliability(self, Ximp=None, alpha=0.95):
        '''
        Get the reliability of imputed entries. The notion of reliability is a relative quantity across all imputed entries.
        Entries with higher reliability are more likely to have small imputation error. 
        '''
        if all(self._cont_indices):
            return self.get_reliability_cont(Ximp, alpha)
        elif all(self._ord_indices):
            return self.get_reliability_ord()
        else:
            raise ValueError('Reliability computation is only available for either all continuous variables or all ordinal variables')


    def sample_imputation(self, X=None, num=5, num_ord_updates=1):
        '''
        Sample multiple imputed datasets using the currently fitted method.
        If X is None, set X as the data used to fit the model.
        Args:
            X: array of shape (n_samples, n_features) or None.
                The dataset to be imputed. Use the seen data for model fitting if None.
            num: int
                The number of imputation samples to draw.
            num_ord_updates: int
                The number of iterations to perform for estimating latent mean at ordinals.
        Return:
            X_imp_num: array of shape (n_samples, n_features, num)
                Imputed dataset.
        '''
        if X is None:
            X = self.transform_function.X

        if all(self._cont_indices):
            Z, Z_ord_lower, Z_ord_upper = self._observed_to_latent(X_to_transform=X)
            Z_imp_num = self._sample_latent(Z=Z, Z_ord_lower=Z_ord_lower, Z_ord_upper=Z_ord_upper, num=num, num_ord_updates=num_ord_updates) 
            X_imp_num = np.zeros_like(Z_imp_num)
            for i in range(num):
                X_imp_num[...,i] = self._latent_to_imp(Z=Z_imp_num[...,i], X_to_impute=X)
        else:
            # slower 
            n, p = X.shape
            X_imp_num = np.empty((n, p, num))
            Z_cont = self.transform_function.get_cont_latent(X_to_transform=X)
            for i in range(num):
                # Z_ord_lower and Z_ord_upper will be different across i
                Z, Z_ord_lower, Z_ord_upper = self._observed_to_latent(X_to_transform=X, Z_cont=Z_cont)
                # TODO: complete Z
                Z_imp = self._sample_latent(Z=Z, Z_ord_lower=Z_ord_lower, Z_ord_upper=Z_ord_upper, num=1, num_ord_updates=num_ord_updates) 
                X_imp_num[...,i] = self._latent_to_imp(Z=Z_imp[...,0], X_to_impute=X)
        return X_imp_num


    def fit_change_point_test(self, X, cont_indices=None, nsamples=100, verbose=False):
        assert self._training_mode == 'minibatch-online'
        X = self._preprocess_data(X, cont_indices)
        self.transform_function = OnlineTransformFunction(self._cont_indices, self._ord_indices, window_size=self._window_size)
        n,p = X.shape
        self._corr = np.identity(p)
        pvals = defaultdict(list)
        test_stats = defaultdict(list)

        i=0
        while True:
            batch_lower= i*self._batch_size
            batch_upper=min((i+1)*self._batch_size, n)
            if batch_lower>=n:
                break 
            if verbose:
                print(f'start batch {i+1}')
            indices = np.arange(batch_lower, batch_upper, 1)
            # Use the first batch to initialize the window. Thus in evaluation the first batch should be ignored.
            if i==0:
                self.transform_function.init_window(X[indices,:])
            _pval, _diff = self.change_point_test(X[indices,:], step_size=self.stepsize(i+1), nsamples=nsamples)
            if nsamples>0:
                for t in self._corr_diff_type:
                    pvals[t].append(_pval[t])
                    test_stats[t].append(_diff[t])
            i+=1
        out = {'pval':pvals, 'statistics':test_stats}
        return out

    ################################################
    #### offline functions
    ################################################

    def fit_transform_offline(self, X, **kwargs):
        '''
        Implement fit_transform when the training mode is 'standard' or 'minibatch-offline'
        '''
        self.fit_offline(X, **kwargs)
        X_imp = self.transform()
        return X_imp

    def fit_offline(self, X, first_fit=True, max_iter=None, convergence_verbose=True):
        '''
        Implement fit when the training mode is 'standard' or 'minibatch-offline'
        '''
        X = self._preprocess_data(X)
        # do marginal estimation
        # for every fit, a brand new marginal transformation is used 
        if first_fit:
            cdf_types, inverse_cdf_types = self.get_cdf_estimation_type(p = X.shape[1])
            self.transform_function = TransformFunction(X, 
                                                        cont_indices=self._cont_indices, 
                                                        ord_indices=self._ord_indices,
                                                        cdf_types=cdf_types,
                                                        inverse_cdf_types=inverse_cdf_types
                                                       )

            Z, Z_ord_lower, Z_ord_upper = self._observed_to_latent()
        else:
            Z_ord_lower, Z_ord_upper = self._Z_ord_lower, self._Z_ord_upper
            Z = self._latent_Zimp.copy()
            Z[np.isnan(X)] = np.nan

        # estimate copula correlation matrix
        Z_imp, C_ord = self._fit_covariance(Z, Z_ord_lower, Z_ord_upper, 
            first_fit=first_fit, max_iter=max_iter, convergence_verbose=convergence_verbose)

        # attributes to store after model fitting
        self._latent_Zimp = Z_imp
        self._latent_Cord = C_ord

        # attributes to store for additional training
        self._Z_ord_lower = Z_ord_lower
        self._Z_ord_upper = Z_ord_upper

    ################################################
    #### online functions 
    ################################################

    def fit_transform_online(self, X, X_true=None, n_train=0):
        '''
        Implement fit_transform when the training mode is 'minibatch-online'
        '''
        if X_true is not None:
            X_true = np.array(X_true)
        cdf_types, inverse_cdf_types = self.get_cdf_estimation_type(p = X.shape[1])
        self.transform_function = OnlineTransformFunction(self._cont_indices, 
                                                          self._ord_indices, 
                                                          window_size=self._window_size, 
                                                          decay = self._decay, 
                                                          cdf_types=cdf_types,
                                                          inverse_cdf_types=inverse_cdf_types
                                                         )
        n,p = X.shape
        X_imp = np.zeros_like(X)
        self._corr = np.identity(p)

        # initialize the model
        n_train = self._batch_size if n_train == 0 else n_train
        assert n_train > 0
        ind_train = np.arange(n_train)
        X_train = X[ind_train] if X_true is None else X_true[ind_train]
        self.transform_function.update_window(X_train)
        _ = self.partial_fit(X_batch = X_train, step_size=1)
        X_imp[ind_train] = self.transform(X = X_train, num_ord_updates=self._num_ord_updates)

        i=0
        while True:
            batch_lower = n_train + i*self._batch_size
            batch_upper = min(n_train + (i+1)*self._batch_size, n)
            if batch_lower>=n:
                break 
            indices = np.arange(batch_lower, batch_upper, 1)
            _X_true = None if X_true is None else X_true[indices]
            X_imp[indices] = self.partial_fit_transform(X[indices], step_size=self.stepsize(i+1), X_true=_X_true)
            i+=1
            if self._verbose > 0:
                print(f'finish batch {i}')
        return X_imp

    def partial_fit_transform(self, X_batch, step_size=0.5, X_true=None):
        """
        Updates the fit of the copula using the data in X_batch and returns the 
        imputed values and the new correlation for the copula

        Args:
            X_batch (matrix): data matrix with entries to use to update copula and be imputed
            step_size (float in (0,1)): tunes how much to weight new covariance estimates
            marginal_update: 
                whether to update the marginal using observation in the new minibatch
            X_true:
                If not None, it indicates that some (could be all) of the missing entries of X_batch are revealed,
                and stored in X_true, after the imputation of X_batch. Those observation entries will be used to 
                update the model. 
        Returns:
            X_imp (matrix): X_batch with missing values imputed
        """
        # impute missing entries in new data using previously fitted model
        # just a step of out-of-sample imputation
        X_for_update = X_batch if X_true is None else X_true

        if self._realtime_marginal:
            X_imp = X_batch.copy()
            for i,x in enumerate(X_batch):
                X_imp[i] = self.transform(X = x.reshape((1, -1)), num_ord_updates = self._num_ord_updates)
                self.transform_function.update_window(X_for_update[i].reshape((1, -1)))
        else:
            X_imp = self.transform(X = X_batch, num_ord_updates=self._num_ord_updates)
            self.transform_function.update_window(X_for_update)

        # use new model to update model parameters
        prev_corr = self._corr.copy()
        new_corr = self.partial_fit(X_batch=X_for_update, step_size=step_size, model_update=True)
        diff = self.get_matrix_diff(prev_corr, self._corr, self._corr_diff_type)
        self._update_corr_diff(diff)
        return X_imp
        
    def partial_fit(self, X_batch, step_size=0.5, model_update=True):
        '''
        Update the copula correlation from new samples in X_batch, with given step size
        '''
        Z_ord_lower, Z_ord_upper = self.transform_function.get_ord_latent(X_to_transform=X_batch)
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper)
        Z_cont = self.transform_function.get_cont_latent(X_to_transform=X_batch) 
        Z = np.empty_like(X_batch)
        Z[:, self._cont_indices] = Z_cont
        Z[:, self._ord_indices] = Z_ord
        corr, Z_imp, Z, C_ord, loglik = self._em_step(Z, Z_ord_lower, Z_ord_upper)

        new_corr = corr*step_size + (1-step_size)*self._corr
        if model_update:
            self._corr = new_corr
            self._latent_Zimp = Z_imp
            self._latent_Cord = C_ord
            self.likelihood.append(loglik)

        return new_corr

    def change_point_test(self, X_batch, step_size, nsamples=100):
        n,p = X_batch.shape
        missing_indices = np.isnan(X_batch)
        prev_corr = self._corr.copy()
        changing_stat = defaultdict(list)

        X_to_impute = np.zeros_like(X_batch) * np.nan
        for i in range(nsamples):
            z = self._rng.multivariate_normal(np.zeros(p), prev_corr, n)
            # mask
            x = np.empty((n,p))
            x[:,self.cont_indices] = self.transform_function.impute_cont_observed(z, X_to_impute)
            x[:,self.ord_indices] = self.transform_function.impute_ord_observed(z, X_to_impute)
            x[missing_indices] = np.nan
            # TODO: compare with enabling marginal_update
            new_corr = self.partial_fit(x, step_size=step_size, model_update=False, marginal_update=False)
            diff = self.get_matrix_diff(prev_corr, new_corr, self._corr_diff_type)
            self._update_corr_diff(diff, output=changing_stat)

        new_corr = self.partial_fit(X_batch, step_size=step_size, model_update=True, marginal_update=True)
        diff = self.get_matrix_diff(prev_corr, new_corr, self._corr_diff_type)
        self._update_corr_diff(diff)

        # compute empirical p-values
        changing_stat = pd.DataFrame(changing_stat)
        pval = {}
        if nsamples>0:
            for t in self._corr_diff_type:
                pval[t] = (np.sum(diff[t]<changing_stat[t])+1)/(nsamples+1)
        return pval, diff

    ################################################
    #### core functions
    ################################################

    def _latent_to_imp(self, Z, X_to_impute=None):
        '''
        Transform the complete latent matrix Z to the observed space, but only keep values at missing entries (to be imputed). 
        All values at observe entries will be replaced with original observation in X_to_impute.
        Args:
            X_to_transform: (nsamples, nfeatures) or None
                If None, self.transform_function.X will be used. 
        '''
        # During the fitting process, all ordinal columns are moved to appear before all continuous columns
        # Rearange the obtained results to go back to the original data ordering
        if X_to_impute is None:
            X_to_impute = self.transform_function.X
        X_imp = X_to_impute.copy()
        if any(self._cont_indices):
            X_imp[:,self._cont_indices] = self.transform_function.impute_cont_observed(Z=Z, X_to_impute=X_to_impute)
        if any(self._ord_indices):
            X_imp[:,self._ord_indices] = self.transform_function.impute_ord_observed(Z=Z, X_to_impute=X_to_impute)
        return X_imp

    def _observed_to_latent(self, X_to_transform=None, Z_cont=None):
        if X_to_transform is None:
            X_to_transform = self.transform_function.X
        if Z_cont is None:
            Z_cont = self.transform_function.get_cont_latent(X_to_transform=X_to_transform)
        Z_ord_lower, Z_ord_upper = self.transform_function.get_ord_latent(X_to_transform=X_to_transform)
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper)
        # Z = np.concatenate((Z_ord, Z_cont), axis=1)
        Z = np.empty_like(X_to_transform)
        Z[:, self.cont_indices] = Z_cont
        Z[:, self.ord_indices] = Z_ord
        return Z, Z_ord_lower, Z_ord_upper

    def _init_copula_corr(self, Z):
        '''
        Initialize the copula correlaiont matrix using incomplete Z. First complete Z and then takes its sample correlaiton matrix.
        '''
        n,p = Z.shape
        # mean impute the missing continuous values for the sake of covariance estimation
        Z_imp = Z.copy()
        Z_imp[np.isnan(Z_imp)] = 0.0

        # initialize the correlation matrix
        self._corr = np.corrcoef(Z_imp, rowvar=False)
        if self._verbose > 1:
            _svdvals = svdvals(self._corr)
            print(f'singular values of the initialized correlation has min {_svdvals.min():.5f} and max {_svdvals.max():.5f}')


    def _fit_covariance(self, Z, Z_ord_lower, Z_ord_upper, first_fit=True, max_iter=None, convergence_verbose=True):
        """
        Fits the gaussian copula correlation matrix using only the transformed data in the latent space.

        Args:
            Z (nsamples, nfeatures)
                Transformed latent matrix
            Z_ord_upper, Z_ord_lower (nsamples, nfeatures_ord)
                Upper and lower bound to sample from
            max_iter: int or None
                The maximum number of iterations to run. If None, use self,_max_iter.
        Returns:
            C_ord
            Z_imp (nsamples, nfeatures)
                The completed matrix in the latent space
        """
        if first_fit:
           self._init_copula_corr(Z)

        n = len(Z)
        # permutation of indices of data for stochastic fitting
        if self._training_mode=='minibatch-offline':
            training_permutation = self._rng.permutation(n)

        # determine the maximal iteraitons to run        
        if self._training_mode=='minibatch-offline' and self._num_pass is not None:
            max_iter = (np.ceil(n/self._batch_size) * self._num_pass).astype(np.int32)
            if self._verbose>0:
                print(f'The number of maximum iteration is set as {max_iter} to have {self._num_pass} passes over all data')
        else:
            max_iter = self._max_iter if max_iter is None else max_iter

        converged = False
        Z_imp = np.empty_like(Z)
        for i in range(max_iter):
            # track the change ratio of copula correlation as stopping criterion
            prev_corr = self._corr
            if np.isnan(prev_corr).any():
                raise ValueError(f'Unexpected nan in updated copula correlation at iteration {i}')

            # run EM iterations
            if self._training_mode == 'standard':
                # standard EM: each iteration uses all data points
                corr, Z_imp, Z, C_ord, iterloglik = self._em_step(Z, Z_ord_lower, Z_ord_upper)
                self._corr = corr
            else:
                # mini-batch EM: more frequent parameter update by using data input with smaller size at each iteration
                batch_lower = (i * self._batch_size) % n
                batch_upper = ((i+1) * self._batch_size) % n
                if batch_upper < batch_lower:
                    # we have wrapped around the dataset
                    indices = np.concatenate((training_permutation[batch_lower:], training_permutation[:batch_upper]))
                else:
                    indices = training_permutation[batch_lower:batch_upper]
                corr, Z_imp_batch, Z_batch, C_ord, iterloglik = self._em_step(Z[indices], Z_ord_lower[indices], Z_ord_upper[indices])
                Z_imp[indices] = Z_imp_batch
                Z[indices] = Z_batch
                step_size = self.stepsize(i+1)
                self._corr = corr*step_size + (1 - step_size)*prev_corr

            self._iter += 1
            # stop if the change in the correlation estimation is below the threshold
            corrudpate = self._get_scaled_diff(prev_corr, self._corr)
            self.corrupdate.append(corrudpate)
            if self._verbose>0:
                print(f"Iteration {self._iter}: copula parameter change {corrudpate:.4f}, likelihood {iterloglik:.4f}")
            
            # append new likelihood and determine if early stopping criterion is satisfied
            converged = self._update_loglikelihood(iterloglik)
                
            if corrudpate < self._threshold:
                converged = True

            if converged:
                break
                
        # store the number of iterations and print if converged
        if convergence_verbose:
            self._set_n_iter(converged, i)
        return  Z_imp, C_ord





    def _em_step(self, Z, r_lower, r_upper):
        """
        Executes one step of the EM algorithm to update the covariance 
        of the copula

        Args:
            Z (matrix): Latent values
            r_lower (matrix): lower bound on latent ordinals
            r_upper (matrix): upper bound on latent ordinals

        Returns:
            sigma (matrix): an estimate of the covariance of the copula
            Z_imp (matrix): estimates of latent values
            Z (matrix): Updated latent values

        """
        n,p = Z.shape
        assert n>0, 'EM step receives empty input'
        max_workers = self._max_workers
        num_ord_updates = self._num_ord_updates

        out_dict = {}
        out_dict['var_ordinal'] = np.zeros((n,p))
        out_dict['Z_imp'] = Z.copy()
        out_dict['Z'] = Z
        out_dict['loglik'] = 0
        out_dict['C'] = np.zeros((p,p))

        has_truncation = self.has_truncation()
        if max_workers ==1:
            args = ('em', Z, r_lower, r_upper, self._corr, num_ord_updates, self._ord_indices, has_truncation)
            res_dict = _latent_operation_body_(args)
            for key in ['Z_imp', 'Z', 'var_ordinal']:
                out_dict[key] = res_dict[key]
            for key in ['loglik', 'C']:
                out_dict[key] += res_dict[key]/n
        else:
            divide = n/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [('em',
                     Z[divide[i]:divide[i+1]].copy(), 
                     r_lower[divide[i]:divide[i+1]], 
                     r_upper[divide[i]:divide[i+1]], 
                     self._corr, 
                     num_ord_updates,
                     self._ord_indices,
                     has_truncation
                    ) for i in range(max_workers)]
            with ProcessPoolExecutor(max_workers=max_workers) as pool: 
                res = pool.map(_latent_operation_body_, args)
                for i, res_dict in enumerate(res):
                    for key in ['Z_imp', 'Z', 'var_ordinal']:
                        out_dict[key][divide[i]:divide[i+1]] = res_dict[key]
                    for key in ['loglik', 'C']:
                        out_dict[key] += res_dict[key]/n

        Z_imp = out_dict['Z_imp']
        C = out_dict['C']
        C_ord = out_dict['var_ordinal']
        sigma = np.cov(Z_imp, rowvar=False) + C 
        try:
            sigma = self._project_to_correlation(sigma)
        except ZeroDivisionError:
            print("unexpected zero covariance for the latent Z")
            _min, _max = C.diagonal().min(), C.diagonal().max()
            print(f'The diagonals of C ranges from min {_min} to max {_max}')
            _m = np.cov(Z_imp, rowvar=False)
            _min, _max = _m.diagonal().min(), _m.diagonal().max()
            print(f'The diagonals of empirical covariance of Z_imp ranges from min {_min} to max {_max}')
            idp = _m.diagonal() == _min
            print(f'Min diagonal appears in {np.flatnonzero(idp)}-th variable with values:')
            print(np.round(Z_imp[:,idp],4))
            print(f'The fitted window is {self.transform_function.X[:,idp]}')
            raise
        return sigma, Z_imp, Z, C_ord, out_dict['loglik']

    def _sample_latent(self, Z, Z_ord_lower, Z_ord_upper, num, num_ord_updates=2):
        '''
        Given incomplete Z, whice has missing entries due to missing observation, fill up those missing entries using the multivariate normal assumption in the latent space.
        '''
        n, p = Z.shape
        max_workers = self._max_workers

        seed = self._sample_seed
        additional_args = {'num':num, 'seed':seed}
        self._sample_seed += 1

        has_truncation = self.has_truncation()
        if max_workers ==1:
            args = ('sample', Z, Z_ord_lower, Z_ord_upper, self._corr, num_ord_updates, self._ord_indices, has_truncation, additional_args)
            res_dict = _latent_operation_body_(args)
            Z_imp_num = res_dict['Z_imp_sample']
        else:
            divide = n/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [('sample',
                     Z[divide[i]:divide[i+1]].copy(), 
                     Z_ord_lower[divide[i]:divide[i+1]], 
                     Z_ord_upper[divide[i]:divide[i+1]], 
                     self._corr, 
                     num_ord_updates,
                     self._ord_indices,
                     has_truncation,
                     additional_args
                    ) for i in range(max_workers)]
            Z_imp_num = np.empty((n,p,num))
            with ProcessPoolExecutor(max_workers=max_workers) as pool: 
                res = pool.map(_latent_operation_body_, args)
                for i, res_dict in enumerate(res):
                    Z_imp_num[divide[i]:divide[i+1],...] = res_dict['Z_imp_sample']
        return Z_imp_num

    def _fillup_latent(self, Z, Z_ord_lower, Z_ord_upper, num_ord_updates=2):
        '''
        Given incomplete Z, whice has missing entries due to missing observation, fill up those missing entries using the multivariate normal assumption in the latent space.
        '''
        n, p = Z.shape
        max_workers = self._max_workers

        has_truncation = self.has_truncation()
        if max_workers ==1:
            args = ('fillup', Z, Z_ord_lower, Z_ord_upper, self._corr, num_ord_updates, self._ord_indices, has_truncation)
            res_dict = _latent_operation_body_(args)
            Z_imp, C_ord = res_dict['Z_imp'], res_dict['var_ordinal']
        else:
            divide = n/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [('fillup',
                     Z[divide[i]:divide[i+1]].copy(), 
                     Z_ord_lower[divide[i]:divide[i+1]], 
                     Z_ord_upper[divide[i]:divide[i+1]], 
                     self._corr, 
                     num_ord_updates,
                     self._ord_indices,
                     has_truncation
                    ) for i in range(max_workers)]
            Z_imp = np.empty((n,p))
            C_ord = np.empty((n,p))
            with ProcessPoolExecutor(max_workers=max_workers) as pool: 
                res = pool.map(_latent_operation_body_, args)
                for i, res_dict in enumerate(res):
                    Z_imp[divide[i]:divide[i+1]] = res_dict['Z_imp']
                    C_ord[divide[i]:divide[i+1]] = res_dict['var_ordinal']
        return Z_imp, C_ord

    def _get_cond_std_missing(self, X=None, Cord=None):
        '''
        The conditional std of each missing location given other observation. 
        '''
        if Cord is None:
            try:
                Cord = self._latent_Cord
            except AttributeError:
                print(f'The model has not been fitted yet. Either fit the model first or supply Cord')
                raise 

        if X is None:
            X = self.transform_function.X

        std_cond = np.zeros_like(X)
        obs_loc = ~np.isnan(X)
        std_cond[obs_loc] = np.nan

        for i,x_row in enumerate(X):
            missing_indices = np.isnan(x_row)
            obs_indices = ~missing_indices

            if any(missing_indices):
                sigma_obs_obs = self._corr[np.ix_(obs_indices,obs_indices)]
                sigma_obs_missing = self._corr[np.ix_(obs_indices, missing_indices)]
                sigma_obs_obs_inv_obs_missing = np.linalg.solve(sigma_obs_obs, sigma_obs_missing)

                # compute quantities
                # _var = 1 - np.diagonal(np.matmul(sigma_obs_missing.T, sigma_obs_obs_inv_obs_missing))
                # use einsum for faster and fewer computation
                _var = 1 - np.einsum('ij, ji -> i', sigma_obs_missing.T, sigma_obs_obs_inv_obs_missing)
                # When there exists valid ordinal observation, we will have self._latent_Cord[i, obs_indices].sum() positive.
                if Cord[i, obs_indices].sum()>0:
                    _var += np.einsum('ij, j, ji -> i', sigma_obs_obs_inv_obs_missing.T, Cord[i, obs_indices], sigma_obs_obs_inv_obs_missing)
                std_cond[i, missing_indices] = np.sqrt(_var)
        return std_cond
    

    def get_reliability_cont(self, Ximp, alpha=0.95):
        '''
        Implements get_reliability when all variabels are continuous.
        '''
        ct = self.get_imputed_confidence_interval(alpha = alpha)
        d = ct['upper'] - ct['lower']

        d_square, x_square = np.power(d,2), np.power(Ximp, 2)
        missing_loc = np.isnan(self.transform_function.X)
        # reliability has np.nan at observation locations because d has np.nan at those locations
        reliability = (d_square[missing_loc].sum() - d_square) / (x_square[missing_loc].sum() - x_square) 
        return reliability

    def get_reliability_ord(self):
        '''
        Implements get_reliability when all variabels are ordinal.
        '''
        std_cond = self._get_cond_std_missing()

        try:
            Zimp = self._latent_Zimp
        except AttributeError:
            print(f'Cannot compute reliability before model fitting and imputation')
            raise 

        Z_ord_lower, _ = self.transform_function.get_ord_latent()

        reliability = np.zeros_like(Zimp) + np.nan
        p = Zimp.shape[1]
        for j in range(p):
            # get cutsoff
            col = Z_ord_lower[:,j]
            missing_indices = np.isnan(col)
            cuts = np.unique(col[~missing_indices])
            cuts = cuts[np.isfinite(cuts)]

            # compute reliability/the probability lower bound
            for i,x in enumerate(missing_indices):
                if x:
                    t = np.abs(Zimp[i,j] - cuts).min()
                    reliability[i,j] = 1 - np.power(std_cond[i,j]/t, 2)
        return reliability

    def _project_to_correlation(self, covariance):
        """
        Projects a covariance to a correlation matrix, normalizing it's diagonal entries. Only checks for diagonal entries to be positive.

        Args:
            covariance (matrix): a covariance matrix

        Returns:
            correlation (matrix): the covariance matrix projected to a correlation matrix
        """
        D = np.diagonal(covariance)
        if any(np.isclose(D, 0)): 
            raise ZeroDivisionError("unexpected zero covariance for the latent Z") 
        D_neg_half = 1.0/np.sqrt(D)
        covariance *= D_neg_half
        return covariance.T * D_neg_half

    def _init_Z_ord(self, Z_ord_lower, Z_ord_upper):
        """
        Initializes the observed latent ordinal values by sampling from a standard
        Gaussian trucated to the inveral of Z_ord_lower, Z_ord_upper

        Args:
            Z_ord_lower (matrix): lower range for ordinals
            Z_ord_upper (matrix): upper range for ordinals

        Returns:
            Z_ord (range): Samples drawn from gaussian truncated between Z_ord_lower and Z_ord_upper
        """
        Z_ord = np.empty(Z_ord_lower.shape)
        Z_ord[:] = np.nan

        n, k = Z_ord.shape
        obs_indices = ~np.isnan(Z_ord_lower)

        u_lower = np.copy(Z_ord_lower)
        u_lower[obs_indices] = norm.cdf(Z_ord_lower[obs_indices])
        u_upper = np.copy(Z_ord_upper)
        u_upper[obs_indices] = norm.cdf(Z_ord_upper[obs_indices])
        assert all(0<=u_lower[obs_indices]) and all(u_lower[obs_indices] <= u_upper[obs_indices]) and  all(u_upper[obs_indices]<=1)

        for i in range(n):
            for j in range(k):
                if not np.isnan(Z_ord_upper[i,j]) and u_upper[i,j] > 0 and u_lower[i,j]<1:
                    u_sample = self._rng.uniform(u_lower[i,j],u_upper[i,j])
                    Z_ord[i,j] = norm.ppf(u_sample)
        return Z_ord

    ################################################
    #### helper functions
    ################################################

    def _get_scaled_diff(self, prev_sigma, sigma):
        """
        Get's the scaled difference between two correlation matrices

        Args:
            prev_sigma (matrix): previous estimate of a matrix
            sigma (matrix): current estimate of a matrix

        Returns: 
            diff (float): scaled distance between the inputs
        """

        return np.linalg.norm(sigma - prev_sigma) / np.linalg.norm(sigma)


    def _preprocess_data(self, X, set_indices = True):
        '''
        Store column names, set continuous/ordinal variable indices and change X to be a numpy array
        '''
        if hasattr(X, 'columns'):
            self.features_names = np.array(X.columns.to_list())
        X = np.asarray(X)
        if set_indices:
            self.set_indices(X)
        return X

    def store_var_type(self, **indices):
        '''
        Store the integer based index for special variable types in self.var_type_dict. 
        '''
        for name, values in indices.items():
            if values is not None:
                self.var_type_dict[name] = values

    def has_truncation(self):
        truncation = False
        for name in ['lower_truncated', 'upper_truncated', 'twosided_truncated']:
            if name in self.var_type_dict:
                truncation = True
                break
        return truncation

    def get_cdf_estimation_type(self, p):
        '''
        Return a list of str indicating the type of cdf estimation using self.var_type_dict
        '''
        cdf_types = np.array(['empirical'] * p, dtype = 'U20')
        inverse_cdf_types = np.array(['empirical'] * p, dtype = 'U20')
        for name, values in self.var_type_dict.items():
            if name in ['lower_truncated', 'upper_truncated', 'twosided_truncated']:
                cdf_types[values] = name
                inverse_cdf_types[values] = name
        return cdf_types, inverse_cdf_types
        
    def set_indices(self, X):
        '''
        set variable types
        '''
        p = X.shape[1]
        # boolean indexing
        var_type = self.get_vartype_indices(X)
        # if there are pre-fixed variable types, modify var_type_list to be consistent
        _merged_var = var_type.copy()
        for name, values in self.var_type_dict.items():
            _merged_var[values] = name
        _diff = _merged_var != var_type
        if any(_diff):
            if self._verbose > 1:
                print('Caution: the user specified variable types differ from the models automatic decision')
                loc = np.flatnonzero(_diff)
                print(f'Conflicts at {loc}: user decision {_merged_var[loc]}, model decision {var_type[loc]}')
        var_type = _merged_var

        # indexing differenting continuous and non-continuous
        self._cont_indices = var_type == 'continuous'
        self._ord_indices = ~self._cont_indices
        # integer based indexing
        self.cont_indices = np.flatnonzero(self._cont_indices)
        self.ord_indices = np.flatnonzero(self._ord_indices)

        # set 
        var_type_dict = defaultdict(list)
        for i,x in enumerate(var_type):
            var_type_dict[x].append(i)

        for key,value in self.var_type_dict.items():
            if key not in var_type_dict:
                raise
            new_value = var_type_dict[key]
            if not set(value).issubset(set(new_value)):
                print(key, set(value), set(new_value))
                raise

        self.var_type_dict = var_type_dict

    def get_vartypes(self, feature_names=None):
        '''
        Return the variable types used during the model fitting. Each variable is one of the following:
            'continuous', 'ordinal', 'lower_truncated', 'upper_truncated', 'twosided_truncated'
        '''
        _var_types = self.var_type_dict.copy()
        if feature_names is not None:
            names = list(feature_names)
            for key,value in _var_types.items():
                _var_types[key] = [names[i] for i in value]
        for name in var_type_names:
            if name not in _var_types:
                _var_types[name] = []
        return _var_types

    def get_vartype_indices(self, X):
        """
        get's the indices of continuos columns by returning
        those indicies which have at least max_ord distinct values

        Args:
            X (matrix): input matrix
            max_ord (int): maximum number of distinct values an ordinal can take on in a column

        Returns:
            indices (array): indices of the columns which have at most max_ord distinct entries
        """
        def is_cont_using_counts(counts):
            return len(counts)>0 and counts.max()/counts.sum() < self._min_ord_ratio

        def is_special_type(x):
            obs = ~np.isnan(x)
            x = x[obs]
            n = len(x)
            uniques, counts = np.unique(x, return_counts=True)
            if len(counts) == 1:
                print('Remove variables with only a single observed level.')
                raise
            #below_max_ord = len(uniques) <= self._max_ord
            is_ord = (counts.max()/n) >= self._min_ord_ratio

            lower_truncated_thre = counts[0]/n >= self._min_ord_ratio
            upper_truncated_thre = counts[-1]/n >= self._min_ord_ratio

            is_lower_truncated = False
            if lower_truncated_thre:
                is_lower_truncated = is_cont_using_counts(counts[1:])

            is_upper_truncated = False
            if upper_truncated_thre:
                is_upper_truncated = is_cont_using_counts(counts[:-1])

            is_twoside_truncated = False
            if lower_truncated_thre and upper_truncated_thre:
                # test if the remaing values could be treated as continuous after removing truncated values
                is_twoside_truncated = is_cont_using_counts(counts[1:-1])

            assert is_twoside_truncated + is_lower_truncated + is_upper_truncated <= 1

            return is_ord, is_lower_truncated, is_upper_truncated, is_twoside_truncated

        def which_type(is_ord, is_lower_truncated, is_upper_truncated, is_twoside_truncated):
            if is_ord:
                if is_lower_truncated:
                    t = 'lower_truncated'
                elif is_upper_truncated:
                    t = 'upper_truncated'
                elif is_twoside_truncated:
                    t = 'twosided_truncated'
                else:
                    t = 'ordinal'
            else:
                t = 'continuous'
            return t


        ord_indices = []
        lower_truncated_indices = []
        upper_truncated_indices = []
        twoside_truncated_indices = []

        var_type_list = []

        for i, col in enumerate(X.T):
            is_ord, is_lower_truncated, is_upper_truncated, is_twoside_truncated = is_special_type(col)
            #ord_indices.append(is_ord)
            #twoside_truncated_indices.append(is_twoside_truncated)
            #lower_truncated_indices.append(is_lower_truncated)
            #upper_truncated_indices.append(is_upper_truncated)
            var_type_list.append(which_type(is_ord, is_lower_truncated, is_upper_truncated, is_twoside_truncated))

        #var_type_dict = {'ordinal': np.array(ord_indices),
        #                'lower_truncated': np.array(lower_truncated_indices),
        #                'upper_truncated': np.array(upper_truncated_indices),
        #                'twosided_truncated': np.array(twoside_truncated_indices)
        #                }

        var_type_list = np.array(var_type_list, dtype = 'U20')
        return var_type_list

    def _update_loglikelihood(self, iterloglik):
        converged = False
        loglik = self.likelihood
        loglik.append(iterloglik)
        if self._training_mode == 'standard' and len(loglik)>1 and loglik[-1] >= loglik[-2]:
            change_ratio = self._get_scaled_diff(loglik[-2], loglik[-1])
            if change_ratio<self._likel_threshold:
                if self._verbose>1: 
                    print(f'early stop because the likelihood increase is below {self._likel_threshold}')
                converged = True
        return converged


    def _set_n_iter(self, converged, i):
        if self._verbose>0: 
            if not converged:
                print("Convergence not achieved at maximum iterations")
            else:
                print(f"Convergence achieved at iteration {i+1}")
        self.n_iter_ = i+1


    def _update_corr_diff(self, corr_diff, output=None):
        if output is None:
            to_append = self.corr_diff
        else:
            # TODO: also check dict names
            assert isinstance(output, dict)
            to_append = output 
        for t in self._corr_diff_type:
            to_append[t].append(corr_diff[t])


    def get_matrix_diff(self, sigma_old, sigma_new, type = ['F', 'S', 'N']):
        '''
        Return the correlation change tracking statistics, as some matrix norm of normalized matrix difference.
        Support three norms currently: 'F' for Frobenius norm, 'S' for spectral norm and 'N' for nuclear norm. 
        User-defined norm can also be used through simple modification.
        Args:
            simga_old: the estimate of copula correlation matrix based on historical data
            sigma_new: the estiamte of copula correlation matrix based on new batch data
            type (a subset of {'F', 'S', 'N'}): the type of matrix norm to use for constructing test statistics. 
        Returns:
            test_stats: a dictionary with (matrix norm type, the test statistics) as (key, value) pair.
        '''
        p = sigma_old.shape[0]
        u, s, vh = np.linalg.svd(sigma_old)
        factor = (u * np.sqrt(1/s) ) @ vh
        diff = factor @ sigma_new @ factor
        test_stats = {}
        if 'F' in type:
            test_stats['F'] = np.linalg.norm(diff-np.identity(p))
        if 'S' in type or 'N' in type:
            _, s, _ = np.linalg.svd(diff)
        if 'S' in type:
            test_stats['S'] = max(abs(s-1))
        if 'N' in type:
            test_stats['N'] = np.sum(abs(s-1))
        return test_stats




