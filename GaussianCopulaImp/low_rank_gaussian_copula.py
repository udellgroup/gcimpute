from .transform_function import TransformFunction
from .gaussian_copula import GaussianCopula
from .embody import _LRGC_em_col_step_body_, _LRGC_em_row_step_body_
from scipy.stats import norm, truncnorm
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor

class LowRankGaussianCopula(GaussianCopula):
    '''
    Low rank Gaussian copula model.
    This class allows to estimate the parameters of a low rank Gaussian copula model from incomplete data, 
    and impute the missing entries using the learned model.
    It is a special case of Gaussian copula model with structured copula correlation matrix: it admits decomposition sigma*I+A*t(A), 
    where A has shape (p,rank) with rank<p. 

    Attributes
    ----------
    cont_indices: ndarray of (n_features,)
        Indication of continuous(True) or oridnal(False) variable decision. 
    n_iter_: int
        Number of iteration rounds that occurred. Will be less than self._max_iter if early stopping criterion was reached.
    likelihood: list of length n_iter_
        The computed pseudo likelihood value at each iteration.
    feature_names: ndarray of shape n_features
        Names of features seen during fit. Defined only when X has feature names that are all strings.

    Methods
    -------
    fit(X)
        fit a Gaussian copula model from incomplete data and then use the fitted model to impute the missing entries.
    fit_transform(X)
        At each sequentially observed data batch, fit a Gaussian copula model from incomplete data and then use the fitted model to impute the missing entries.
    get_params()
        Get parameters for this estimator.
    get_imputed_confidence_interval(alpha=0.95)
        Get the confidence intervals for the imputed missing entries when all variables are continuous
    get_reliability(Ximp=None, alpha=0.95)
        Get the reliability, a relative quantity across all imputed entries, when either all variables are continuous or all variables are ordinal 
    '''
    def __init__(self, rank, min_ord_ratio=0.1, tol=0.001, likelihood_min_increase=0.01, max_iter=50, random_state=101, n_jobs=1, verbose=0, num_ord_updates=1):
        '''
        Parameters:
            rank: int
                The number of the latent factors, i.e. the rank of the latent data generating space
            min_ord_ratio: float, default=0.1
                When cont_indices is None, variables whose largest occurence ratio among unique values is smaller than min_ord_ratio 
                are regarded as continuous variables.
            tol: float, default=0.01
                The convergence threshold. EM iterations will stop when the parameter update ratio is below this threshold.
            likelihood_min_increase: float, default=0.01
                The minimal likelihood increase ratio required to keep running the EM algorithm.
            max_iter: int, default=100
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
        '''
        super().__init__(training_mode='standard', min_ord_ratio=min_ord_ratio, tol=tol, max_iter=max_iter, random_state=random_state, n_jobs=n_jobs, verbose=verbose, num_ord_updates=num_ord_updates)
        self._rank = rank
        self._W = None
        self._sigma = None

    def get_params(self):
        '''
        Get parameters for this estimator.

        Returns:
            params: dict
        '''
        # During the fitting process, all ordinal columns are moved to appear before all continuous columns
        # Rearange the obtained results to go back to the original data ordering
        _order = self.back_to_original_order()
        params = {'copula_factor_loading': self._W[_order], 'copula_noise_ratio':self._sigma}
        return params


    def _get_cond_std_missing(self):
        '''
        The conditional std of each missing location given other observation. 
        The computation under LRGC is adjusted by exploting the SVD decomposition of the copula parameter W.
        '''
        try:
            Cord = self._latent_Cord
        except AttributeError:
            print(f'Cannot compute conditional std of missing entries before model fitting and imputation')
            raise 

        std_cond = np.zeros_like(self.transform_function.X)
        obs_loc = ~np.isnan(self.transform_function.X)
        std_cond[obs_loc] = np.nan

        U,d,_ = np.linalg.svd(self._W, full_matrices=False)

        for i,x_row in enumerate(self.transform_function.X):
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

    def _fit_covariance(self, Z, Z_ord_lower, Z_ord_upper):
        """
        See the doc for _fit_covariance in class GaussianCopula()
        """

        # Refine Z_imp using truncated (low-rank) SVD for missing entries to obtain initial parameter estimate
        Z_imp = Z.copy()
        Z_imp = self._init_impute_svd(Z_imp, self._rank, Z_ord_lower, Z_ord_upper)
        # Form latent variable matrix
        # Update entries at obseved ordinal locations from SVD initialization
        if any(self.ord_indices):
            obs_ord = ~np.isnan(Z_ord_lower)
            num_ord = sum(self.ord_indices)
            Z[:,:num_ord][obs_ord] = Z_imp[:,:num_ord][obs_ord].copy()

        # initialize the parameter estimate 
        corr = np.corrcoef(Z_imp, rowvar=False)
        u,d,_ = np.linalg.svd(corr, full_matrices=False)
        sigma = np.mean(d[self._rank:])
        W = u[:,:self._rank] * (np.sqrt(d[:self._rank] - sigma))
        self._W, self._sigma = self._scale_corr(W, sigma)
        if self._verbose>0:
            print(f'Ater initialization, W has shape {self._W.shape} and sigma is {self._sigma:.4f}')

        converged = False
        for i in range(self._max_iter):
            prev_W = self._W

            # run EM iteration
            W_new, sigma_new, Z, C, iterloglik = self._em_step(Z, Z_ord_lower, Z_ord_upper) 
            self._W, self._sigma = W_new, sigma_new

            # stop if the change in the parameter estimation is below the threshold
            wupdate = self._get_scaled_diff(prev_W, self._W)
            if self._verbose>0:
                print(f'Interation {i+1}: noise ratio estimate {self._sigma:.4f}, copula parameter update ratio {wupdate:.4f}, likelihood {iterloglik:.4f}')

            # append new likelihood and determine if early stopping criterion is satisfied
            converged = self._update_loglikelihood(iterloglik)

            if wupdate < self._threshold:
                converged = True
            
            if converged:
                break

        # store the number of iterations and print if converged
        self._set_n_iter(converged, i)

        S = self._comp_S(Z) # re-estimate S to ensure numerical stability
        Z_imp = self._impute(Z, S, self._W)
        return Z_imp, C


    def _comp_S(self, Z):
        """
        Intermidiate step.
        It seems that S must be updated using the last obtained W and sigma, otherwise numerical instability happens. 
        Such problem is not seen in R implementation. Keep an eye.
        Args:
            Z (matrix): the transformed value, at observed continuous entry; the conditional mean, at observed ordinal entry; NA elsewhere
        Returns:
            S: a factor used for imputation
        """
        W, sigma = self._W, self._sigma
        n, k = Z.shape[0], W.shape[1]
        U, d, _ = np.linalg.svd(W, full_matrices=False)
        S = np.zeros((n,k))
        for i in range(n):
            obs_indices = np.nonzero(~np.isnan(Z[i,:]))[0]

            zi_obs = Z[i,obs_indices]
            Ui_obs = U[obs_indices,:]
            UU_obs = np.dot(Ui_obs.T, Ui_obs) # YX: better edit to avoid vector-vector inner product

            S[i,:] = np.linalg.solve(UU_obs + sigma * np.diag(1.0/np.square(d)), np.dot(Ui_obs.T, zi_obs))
        return S

    def _impute(self, Z, S, W):
        """
        Impute missing values
        Args:
            W (matrix): an estimate of the latent coefficient matrix of the low rank Gaussian copula
            sigma (scalar): an estimate of the latent noise variance of the low rank Gaussian copula
            Z (matrix): the transformed value, at observed continuous entry; the conditional mean, at observed ordinal entry; NA elsewhere
            S: a factor used for imputation
        Returns:
            Zimp (matrix): a copy of Z, but with missing entries replaced by their conditional mean imputation.
        """
        n,p = Z.shape
        Zimp = np.copy(Z)
        U,d,_ = np.linalg.svd(W, full_matrices=False)
        for i in range(n):
            index_m = np.nonzero(np.isnan(Z[i,:]))
            Zimp[i,index_m] =  np.dot(U[index_m,:], S[i,:])
        return Zimp

    def _em_step(self, Z, r_lower, r_upper):
        n,p = Z.shape
        assert len(self._W.shape)==2, f'invalid W shape {self._W.shape}'
        assert n>0, 'EM step receives empty input'
        W, sigma = self._W, self._sigma
        max_workers = self._max_workers
        num_ord_updates = self._num_ord_updates

        rank = W.shape[1]
        num_ord = r_lower.shape[1]
        U,d,V = np.linalg.svd(W, full_matrices=False)

        if max_workers == 1:
            args = (Z, r_lower, r_upper, U, d, sigma, num_ord_updates)
            Z, A, S, SS, C_ord, loglik, s_row = _LRGC_em_row_step_body_(args)
            args = (Z, C_ord, U, sigma, A, S, SS)
            W_new, s_col = _LRGC_em_col_step_body_(args)
        else:
            # computation across rows
            divide = n/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [(
                    Z[divide[i]:divide[i+1],:].copy(), 
                    r_lower[divide[i]:divide[i+1],:], 
                    r_upper[divide[i]:divide[i+1],:], 
                    U, d, sigma, num_ord_updates
                    ) for i in range(max_workers)]
            A = np.zeros((n,rank,rank))
            S = np.zeros((n,rank))
            SS = np.zeros_like(A)
            C_ord = np.zeros_like(Z)
            loglik = 0
            s_row = 0
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_LRGC_em_row_step_body_, args)
                for i,(zi, Ai, si, ssi, C_ord_i, loglik_i, s_row_i) in enumerate(res):
                    Z[divide[i]:divide[i+1]] = zi
                    A[divide[i]:divide[i+1]] = Ai
                    S[divide[i]:divide[i+1]] = si
                    SS[divide[i]:divide[i+1]] = ssi
                    C_ord[divide[i]:divide[i+1]] = C_ord_i
                    loglik += loglik_i
                    s_row += s_row_i
            # computation across columns
            divide = p/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [(
                    Z[:,divide[i]:divide[i+1]].copy(), 
                    C_ord[:,divide[i]:divide[i+1]],
                    U[divide[i]:divide[i+1]],
                    sigma,
                    A, S, SS, 
                    ) for i in range(max_workers)]
            W_new = np.zeros_like(U)
            s_col = 0
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_LRGC_em_col_step_body_, args)
                for i,(wnew_i, s_col_i) in enumerate(res):
                    W_new[divide[i]:divide[i+1]] = wnew_i
                    s_col += s_col_i

        s = s_row+np.sum(C_ord)-s_col
        sigma_new = s/float(np.sum(~np.isnan(Z)))
        W_new = np.dot(W_new * d, V)
        W, sigma = self._scale_corr(W_new, sigma_new)
        return W, sigma, Z, C_ord, loglik/n



    def _init_impute_svd(self, Z, rank, Z_ord_lower, Z_ord_upper):
        # first zero initialization on missing entries to obtain SVD
        # then SVD initialization replace zero initialization
        Z_imp = np.copy(Z)
        Z_imp[np.isnan(Z_imp)] = 0.0

        u,s,vh = np.linalg.svd(Z_imp, full_matrices=False)
        u_low_rank = u[:,:rank]
        s_low_rank = s[:rank]
        vh_low_rank = vh[:rank,:]
        Z_imp = np.dot(u_low_rank * s_low_rank, vh_low_rank) 

        k,p = Z_ord_lower.shape[1], Z.shape[1]

        for j in range(k):
            # index is a subset of observed indices
            # pay attension to missing values
            index_o = np.nonzero(~np.isnan(Z[:,j]))[0]
            index = (Z_imp[index_o,j] > Z_ord_upper[index_o,j]) | (Z_imp[index_o,j] < Z_ord_lower[index_o,j])
            Z_imp[index_o[index],j] = Z[index_o[index],j]

        for j in range(k,p,1):
            index_o = ~np.isnan(Z[:,j])
            Z_imp[index_o,j] = Z[index_o,j]

        return Z_imp


    def _scale_corr(self, W, sigma):
        p = W.shape[0]
        tr = np.sum(np.square(W), axis=1)
        sigma = np.mean(1.0/(tr + sigma)) * sigma
        for j in range(p):
            W[j,:] = np.sqrt(1 - sigma) * W[j,:]/np.sqrt(tr[j])
        return W, sigma


    def _impute_missing_oracle(self, X, W, sigma, f = None, finv = None, max_ord_levels = 20):
        # only for continuous matrix
        n, k = X.shape[0], W.shape[1]
        cont_indices = self.get_cont_indices(X, max_ord_levels=max_ord_levels) 
        ord_indices = ~cont_indices
        self.transform_function = TransformFunction(X, cont_indices, ord_indices)
        Z = self.transform_function.get_cont_latent()
        #Z = X
        U, d, _ = np.linalg.svd(W, full_matrices=False)
        S = np.zeros((n,k))
        for i in range(n):
            obs_indices = np.nonzero(~np.isnan(Z[i,:]))[0]

            zi_obs = Z[i,obs_indices]
            Ui_obs = U[obs_indices,:]
            UU_obs = np.dot(Ui_obs.T, Ui_obs) # YX: better edit to avoid vector-vector inner product

            S[i,:] = np.linalg.solve(UU_obs + sigma * np.diag(1.0/np.square(d)), np.dot(Ui_obs.T, zi_obs))

        S = self._comp_S(Z, W, sigma)
        Z_imp = self._impute(Z, S, W)
        X_imp = np.empty(X.shape)
        X_imp = self.transform_function.impute_cont_observed(Z_imp)
        return X_imp








