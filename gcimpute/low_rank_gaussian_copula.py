from .transform_function import TransformFunction
from .gaussian_copula import GaussianCopula
from .embody import _LRGC_em_col_step_body_, _LRGC_latent_operation_body_
from scipy.stats import norm, truncnorm
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor
from .helper_evaluation import grassman_dist

class LowRankGaussianCopula(GaussianCopula):
    '''
    Low rank Gaussian copula model.
    This class allows to estimate the parameters of a low rank Gaussian copula model from incomplete data, 
    and impute the missing entries using the learned model.
    It is a special case of Gaussian copula model with structured copula correlation matrix: it admits decomposition sigma*I+A*t(A), 
    where A has shape (p,rank) with rank<p. 

    Parameters
    ----------
    rank: int
        The number of the latent factors, i.e. the rank of the latent data generating space
    tol: float, default=0.01
        The convergence threshold. EM iterations will stop when the parameter update ratio is below this threshold.
    max_iter: int, default=50
        The number of EM iterations to perform.
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
    min_ord_ratio: float, default=0.1
        Used for automatic variable type decision. The largest mode frequency for continuous variables.



    Methods
    -------
    fit(X)
        Fit a Gaussian copula model on X.
    transform(X)
        Return the imputed X using the stored model.
    fit_transform(X)
        Fit a Gaussian copula model on X and return the transformed X.
    fit_transform_evaluate(X, eval_func)
        Conduct eval_func on the imputed datasets returned after each iteration.
    sample_imputation(X)
        Return multiple imputed datasets X using the stored model.
    get_params()
        Get parameters for this estimator.
    get_vartypes()
        Get the specified variable types used in model fitting.
    get_imputed_confidence_interval()
        Get the confidence intervals for the imputed missing entries.
    get_reliability()
        Get the reliability, a relative quantity across all imputed entries, when either all variables are continuous or all variables are ordinal 
    '''
    def __init__(self, rank, **kwargs):
        super().__init__(training_mode='standard',  **kwargs)
        self._rank = rank
        self._W = None
        self._sigma = None

        #self.corrupdate_grassman = []

    ################################################
    #### public functions
    ################################################

    def get_params(self):
        '''
        Get parameters for this estimator.

        Returns:
            params: dict
        '''
        params = {'copula_factor_loading': self._W, 'copula_noise_ratio':self._sigma}
        return params

    ################################################
    #### core functions
    ################################################

    def _get_cond_std_missing(self, X=None, Cord=None):
        '''
        Specialized implementation for LRGC
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

        U,d,_ = np.linalg.svd(self._W, full_matrices=False)

        for i,x_row in enumerate(X):
            missing_indices = np.isnan(x_row)
            obs_indices = ~missing_indices

            if any(missing_indices):
                Ui_obs = U[obs_indices]
                # Ui_mis has dimension num_mis*k
                Ui_mis = U[missing_indices]
                # dUmis has dimension k*num_mis
                dUmis = np.linalg.solve(np.diag(self._sigma*np.power(d, -2))+np.matmul(Ui_obs.T, Ui_obs), Ui_mis.T)

                _var = self._sigma * (1 + np.einsum('ij, ji -> i', Ui_mis, dUmis))
                if self._latent_Cord[i, obs_indices].sum()>0:
                    # dimension of num_obs*num_mis
                    Wobs_Mobs_inv_WmisT = np.matmul(Ui_obs, dUmis)
                    _var += np.einsum('ij, j, ji -> i', Wobs_Mobs_inv_WmisT.T, Cord[i, obs_indices], Wobs_Mobs_inv_WmisT)
                std_cond[i, missing_indices] = np.sqrt(_var)
        return std_cond

    def _init_copula_corr(self, Z, Z_ord_lower, Z_ord_upper):
        '''
        Specialized implementation for LRGC

        Parameters
        ----------
            Z : array-like of shape (nsamples, nfeatures_ordinal)
                latent matrix 
            Z_ord_lower : array-like of shape (nsamples, nfeatures_ordinal)
                lower range for ordinals
            Z_ord_upper : array-like of shape (nsamples, nfeatures_ordinal)
                upper range for ordinals

        Returns
        -------
            Z_imp: array-like of shape (nsamples, nfeatures)
                The imputed latent values used to initialize the copula correlation 
        '''
        # Refine Z_imp using truncated (low-rank) SVD for missing entries to obtain initial parameter estimate
        Z_imp = Z.copy()
        Z_imp = self._init_impute_svd(Z_imp, self._rank, Z_ord_lower, Z_ord_upper)

        # initialize the parameter estimate 
        corr = np.corrcoef(Z_imp, rowvar=False)
        u,d,_ = np.linalg.svd(corr, full_matrices=False)
        sigma = np.mean(d[self._rank:])
        W = u[:,:self._rank] * (np.sqrt(d[:self._rank] - sigma))
        self._W, self._sigma = self._scale_corr(W, sigma)

        if self._verbose>1:
            print(f'Ater initialization, W has shape {self._W.shape} and sigma is {self._sigma:.4f}')

        return Z_imp

    def _fit_covariance(self, Z, Z_ord_lower, Z_ord_upper, first_fit=True, max_iter=None, convergence_verbose=True):
        """
        See the doc for _fit_covariance in class GaussianCopula()
        """
        if self._rank >= Z.shape[1]:
            raise ValueError('The provided rank must be smaller than the data dimension')
        if first_fit:
            Z_imp = self._init_copula_corr(Z, Z_ord_lower, Z_ord_upper)
            # Form latent variable matrix: Update entries at obseved ordinal locations from SVD initialization
            if any(self._ord_indices):
                Z_ord = Z[:, self._ord_indices].copy()
                obs_ord = ~np.isnan(Z_ord)
                Z_ord[obs_ord] = Z_imp[:,self._ord_indices][obs_ord].copy()
                Z[:, self._ord_indices] = Z_ord

        max_iter = self._max_iter if max_iter is None else max_iter

        converged = False
        for i in range(max_iter):
            prev_W = self._W

            # run EM iteration
            W_new, sigma_new, Z, C, iterloglik = self._em_step(Z, Z_ord_lower, Z_ord_upper) 
            self._W, self._sigma = W_new, sigma_new

            self._iter += 1
            # stop if the change in the parameter estimation is below the threshold
            wupdate = self._get_scaled_diff(prev_W, self._W)
            self.corrupdate.append(wupdate)
            #self.corrupdate_grassman.append(grassman_dist(prev_W, self._W)[0])
            if self._verbose>0:
                print(f'Interation {self._iter}: noise ratio {self._sigma:.4f}, copula parameter change {wupdate:.4f}, likelihood {iterloglik:.4f}')

            # append new likelihood
            self.likelihood.append(iterloglik)

            if wupdate < self._threshold:
                converged = True
            
            if converged:
                break

        # store the number of iterations and print if converged
        if convergence_verbose:
            self._set_n_iter(converged, i)

        Z_imp = self._impute(Z, self._W, self._sigma)
        return Z_imp, C

    def _impute(self, Z, W, sigma):
        """
        Impute missing values in the latent space using provided model parameters

        Parameters
        ----------
            W : array-like of shape (nfeatures, rank)
                the latent coefficient matrix of the low rank Gaussian copula
            sigma : float in (0,1)
                the latent noise variance of the low rank Gaussian copula
            Z : array-like of shape (nsamples, nfeatures)
                the transformed value in the latent space

        Returns
        -------
            Zimp : array-like of shape (nsamples, nfeatures)
                a copy of Z, but with missing entries replaced by their conditional mean imputation.
        """
        n,p = Z.shape
        Zimp = np.copy(Z)
        U,d,_ = np.linalg.svd(W, full_matrices=False)
        for i in range(n):
            #index_m = np.nonzero(np.isnan(Z[i]))
            index_m = np.isnan(Z[i])
            obs_indices = ~index_m

            zi_obs = Z[i,obs_indices]
            Ui_obs = U[obs_indices]
            UU_obs = np.dot(Ui_obs.T, Ui_obs) 

            s = np.linalg.solve(UU_obs + sigma * np.diag(1.0/np.square(d)), np.dot(Ui_obs.T, zi_obs))

            Zimp[i,index_m] =  np.dot(U[index_m], s)
        return Zimp

    def _em_step(self, Z, r_lower, r_upper):
        '''
        Specialized implementation for LRGC
        '''
        n,p = Z.shape
        assert len(self._W.shape)==2, f'invalid W shape {self._W.shape}'
        assert n>0, 'EM step receives empty input'
        W, sigma = self._W, self._sigma
        max_workers = self._max_workers
        num_ord_updates = self._num_ord_updates

        rank = W.shape[1]
        num_ord = r_lower.shape[1]
        U,d,V = np.linalg.svd(W, full_matrices=False)

        out_dict = {}
        out_dict['var_ordinal'] = np.zeros((n,p))
        out_dict['Z'] = Z
        out_dict['loglik'] = 0
        out_dict['A'] = np.zeros((n, rank, rank))
        out_dict['s'] = np.zeros((n, rank))
        out_dict['ss'] = np.zeros((n, rank, rank))
        out_dict['zobs_norm'] = 0

        has_truncation = self.has_truncation()
        if max_workers == 1:
            args = ('em', Z, r_lower, r_upper, U, d, sigma, num_ord_updates, self._ord_indices, has_truncation)
            res_dict = _LRGC_latent_operation_body_(args)
            for key in ['Z', 'var_ordinal', 'A', 's', 'ss']:
                out_dict[key] = res_dict[key]
            for key in ['loglik', 'zobs_norm']:
                out_dict[key] += res_dict[key]
            args = (out_dict['Z'], out_dict['var_ordinal'], U, sigma, out_dict['A'], out_dict['s'], out_dict['ss'])
            W_new, s_col = _LRGC_em_col_step_body_(args)
        else:
            # computation across rows
            divide = n/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [('em',
                    Z[divide[i]:divide[i+1]].copy(), 
                    r_lower[divide[i]:divide[i+1]], 
                    r_upper[divide[i]:divide[i+1]], 
                    U, 
                    d, 
                    sigma, 
                    num_ord_updates,
                    self._ord_indices,
                    has_truncation
                    ) for i in range(max_workers)]
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_LRGC_latent_operation_body_, args)
                for i, res_dict in enumerate(res):
                    for key in ['Z', 'var_ordinal', 'A', 's', 'ss']:
                        out_dict[key][divide[i]:divide[i+1]] = res_dict[key]
                    for key in ['loglik', 'zobs_norm']:
                        out_dict[key] += res_dict[key]

            # computation across columns
            divide = p/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [(
                    out_dict['Z'][:,divide[i]:divide[i+1]].copy(), 
                    out_dict['var_ordinal'][:,divide[i]:divide[i+1]],
                    U[divide[i]:divide[i+1]],
                    sigma,
                    out_dict['A'], 
                    out_dict['s'], 
                    out_dict['ss'], 
                    ) for i in range(max_workers)]
            W_new = np.zeros_like(U)
            s_col = 0
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_LRGC_em_col_step_body_, args)
                for i,(wnew_i, s_col_i) in enumerate(res):
                    W_new[divide[i]:divide[i+1]] = wnew_i
                    s_col += s_col_i

        C_ord = out_dict['var_ordinal']
        Z = out_dict['Z']
        s = out_dict['zobs_norm']+C_ord.sum()-s_col
        sigma_new = s/float(np.sum(~np.isnan(Z)))
        W_new = np.dot(W_new * d, V)
        W, sigma = self._scale_corr(W_new, sigma_new)
        loglik = out_dict['loglik']/n
        return W, sigma, Z, C_ord, loglik

    def _sample_latent(self, Z, Z_ord_lower, Z_ord_upper, num, num_ord_updates=2):
        '''
        Specialized implementation for LRGC
        '''
        n,p = Z.shape
        W, sigma = self._W, self._sigma
        max_workers = self._max_workers
        num_ord_updates = self._num_ord_updates

        rank = W.shape[1]
        num_ord = r_lower.shape[1]
        U,d,V = np.linalg.svd(W, full_matrices=False)

        seed = self._sample_seed
        additional_args = {'num':num, 'seed':seed}
        self._sample_seed += 1

        has_truncation = self.has_truncation()
        if max_workers ==1:
            args = ('sample', Z, Z_ord_lower, Z_ord_upper, U, d, sigma, num_ord_updates, self._ord_indices, has_truncation, additional_args)
            res_dict = _LRGC_latent_operation_body_(args)
            Z_imp_num = res_dict['Z_imp_sample']
        else:
            divide = n/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [('sample',
                     Z[divide[i]:divide[i+1]].copy(), 
                     Z_ord_lower[divide[i]:divide[i+1]], 
                     Z_ord_upper[divide[i]:divide[i+1]], 
                     U,
                     d,
                     sigma,
                     num_ord_updates,
                     self._ord_indices,
                     has_truncation,
                     additional_args
                    ) for i in range(max_workers)]
            Z_imp_num = np.empty((n,p,num))
            with ProcessPoolExecutor(max_workers=max_workers) as pool: 
                res = pool.map(_LRGC_latent_operation_body_, args)
                for i, res_dict in enumerate(res):
                    Z_imp_num[divide[i]:divide[i+1],...] = res_dict['Z_imp_sample']
        return Z_imp_num

    def _fillup_latent(self, Z, Z_ord_lower, Z_ord_upper, num_ord_updates=2):
        '''
        Specialized implementation for LRGC
        '''
        n,p = Z.shape
        W, sigma = self._W, self._sigma
        max_workers = self._max_workers
        num_ord_updates = self._num_ord_updates

        rank = W.shape[1]
        num_ord = Z_ord_lower.shape[1]
        U,d,V = np.linalg.svd(W, full_matrices=False)

        has_truncation = self.has_truncation()
        if max_workers ==1:
            args = ('fillup', Z, Z_ord_lower, Z_ord_upper, U, d, sigma, num_ord_updates, self._ord_indices, has_truncation)
            res_dict = _LRGC_latent_operation_body_(args)
            Z_imp, C_ord = res_dict['Z_imp'], res_dict['var_ordinal']
        else:
            divide = n/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [('fillup',
                     Z[divide[i]:divide[i+1]].copy(), 
                     Z_ord_lower[divide[i]:divide[i+1]], 
                     Z_ord_upper[divide[i]:divide[i+1]], 
                     U,
                     d,
                     sigma,
                     num_ord_updates,
                     self._ord_indices,
                     has_truncation,
                    ) for i in range(max_workers)]
            Z_imp = np.empty((n,p))
            C_ord = np.empty((n,p))
            with ProcessPoolExecutor(max_workers=max_workers) as pool: 
                res = pool.map(_LRGC_latent_operation_body_, args)
                for i, res_dict in enumerate(res):
                    Z_imp[divide[i]:divide[i+1]] = res_dict['Z_imp']
                    C_ord[divide[i]:divide[i+1]] = res_dict['var_ordinal']
        return Z_imp, C_ord

    def _init_impute_svd(self, Z, rank, Z_ord_lower, Z_ord_upper):
        '''
        Return an imputed Z using SVD on the zero-imputed Z

        Parameters
        ----------
            Z : array-like of shape (nsamples, nfeatures_ordinal)
                latent matrix 
            Z_ord_lower : array-like of shape (nsamples, nfeatures_ordinal)
                lower range for ordinals
            Z_ord_upper : array-like of shape (nsamples, nfeatures_ordinal)
                upper range for ordinals
            rank : int
                rank to use for SVD

        Returns
        -------
            Z_imp: array-like of shape (nsamples, nfeatures)
                The imputed latent values used to initialize the copula correlation 
        '''
        Z_imp = np.copy(Z)
        Z_imp[np.isnan(Z_imp)] = 0.0

        u,s,vh = np.linalg.svd(Z_imp, full_matrices=False)
        u_low_rank = u[:,:rank]
        s_low_rank = s[:rank]
        vh_low_rank = vh[:rank,:]
        Z_imp = np.dot(u_low_rank * s_low_rank, vh_low_rank) 

        k,p = Z_ord_lower.shape[1], Z.shape[1]

        j_in_ord = 0
        for j, (_cont, _ord) in enumerate(zip(self._cont_indices, self._ord_indices)):
            if _cont and _ord:
                raise 'Some variable is specified as both continuous and ordinal'
            if not _cont and not _ord:
                raise 'Some variable is specified as neither continuous nor ordinal'
            if _cont:
                index_o = ~np.isnan(Z[:,j])
                Z_imp[index_o,j] = Z[index_o,j]
            if _ord:
                index_o = np.flatnonzero(~np.isnan(Z[:,j]))
                index = (Z_imp[index_o,j] > Z_ord_upper[index_o,j_in_ord]) | (Z_imp[index_o,j] < Z_ord_lower[index_o,j_in_ord])
                Z_imp[index_o[index],j] = Z[index_o[index],j]
                j_in_ord += 1

        return Z_imp

    def _scale_corr(self, W, sigma):
        '''
        Scale W and sigma so that WW^T + sigma I is a correlation matrix
        
        Parameters
        ----------
            W : array-like of shape (nfeatures, rank)
                the latent coefficient matrix of the low rank Gaussian copula
            sigma : float in (0,1)
                the latent noise variance of the low rank Gaussian copula

        Returns
        -------
            W : array-like of shape (nfeatures, rank)
                the adjusted latent coefficient matrix of the low rank Gaussian copula
            sigma : float in (0,1)
                the adjusted latent noise variance of the low rank Gaussian copula
        '''
        p = W.shape[0]
        tr = np.sum(np.square(W), axis=1)
        sigma = np.mean(1.0/(tr + sigma)) * sigma
        for j in range(p):
            W[j,:] = np.sqrt(1 - sigma) * W[j,:]/np.sqrt(tr[j])
        return W, sigma
