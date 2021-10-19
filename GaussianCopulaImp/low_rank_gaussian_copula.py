from .transform_function import TransformFunction
from .gaussian_copula import GaussianCopula
from scipy.stats import norm, truncnorm
import numpy as np


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
    pseudo_likelihood: list of length n_iter_
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
    def __init__(self, rank, cont_indices=None, max_ord=20, min_ord_ratio=0.1, tol=0.001, max_iter=50, random_state=101, n_jobs=1, verbose=0, num_ord_updates=1):
        '''
        Parameters:
            rank: int
                The number of the latent factors, i.e. the rank of the latent data generating space
            cont_indices: list of bool or None, default=None
                The indication of whether a variable should be treated as continuous(True) or ordinal(False) if not None. If None,
                the decision will be decided based on max_ord and min_ord_ratio. 
            max_ord: int, default=20
                When cont_indices is None, variables whose number of unqiue observed values is larger than max_ord are regarded 
                as continuous variables.
            min_ord_ratio: float, default=0.1
                When cont_indices is None, variables whose largest occurence ratio among unique values is smaller than min_ord_ratio 
                are regarded as continuous variables.
            tol: float, default=0.01
                The convergence threshold. EM iterations will stop when the parameter update ratio is below this threshold.
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
        super().__init__(training_mode='standard', cont_indices=cont_indices, max_ord=max_ord, min_ord_ratio=min_ord_ratio, tol=tol, max_iter=max_iter, random_state=random_state, n_jobs=n_jobs, verbose=verbose, num_ord_updates=num_ord_updates)
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

    def fit_offline(self,X):
        '''
        Implement fit for LRGC
        '''
        if self.cont_indices is None:
            self.cont_indices = self.get_cont_indices(X)
            self.ord_indices = ~self.cont_indices

        self.transform_function = TransformFunction(X, self.cont_indices, self.ord_indices)
        # TO DO: consider the order of W
        Z, C = self._fit_covariance(X)
        S = self._comp_S(Z) # re-estimate S to ensure numerical stability
        Z_imp = self._impute(Z, S, self._W)
        # 
        self._latent_Zimp = Z_imp
        self._latent_Cord = C

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

    def _fit_covariance(self, X):
        """
        Estimate the covariance parameters of the low rank Gaussian copula, W and sigma, 
        using the data in X and return the estimates and related quantity. 
        Different from the full rank case, imputation is not conducted in this step.

        Args:
            X (matrix): data matrix with entries to be imputed
        Returns:
            Z (matrix): the transformed value, at observed continuous entry; the conditional mean, at observed ordinal entry; NA elsewhere
            C (matrix): 0 at observed continuous entry; the conditional variance, at observed ordinal entry; NA elsewhere
        """
        Z_ord_lower, Z_ord_upper = self.transform_function.get_ord_latent()
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper)
        Z_cont = self.transform_function.get_cont_latent()
        Z = np.concatenate((Z_ord, Z_cont), axis=1)

        # Initialize Z_imp using truncated (low-rank) SVD for missing entries
        # to obtain initial parameter estimate
        Z_imp = self._init_impute_svd(Z, self._rank, Z_ord_lower, Z_ord_upper)
        corr = np.corrcoef(Z_imp, rowvar=False)
        u,d,_ = np.linalg.svd(corr, full_matrices=False)
        sigma = np.mean(d[self._rank:])
        W = u[:,:self._rank] * (np.sqrt(d[:self._rank] - sigma))
        self._W, self._sigma = self._scale_corr(W, sigma)
        if self._verbose>0:
            print(f'Ater initialization, W has shape {self._W.shape} and sigma is {self._sigma}')

        # Update entries at obseved ordinal locations from SVD initialization
        if sum(self.ord_indices)>0:
            Z_ord[~np.isnan(Z_ord)] = Z_imp[:,:sum(self.ord_indices)][~np.isnan(Z_ord)]
            Z = np.concatenate((Z_ord, Z_cont), axis=1)

        loglik = self.pseudo_likelihood
        for i in range(self._max_iter):
            prev_W = self._W
            W_new, sigma_new, Z, C, iterloglik = self._em_step(Z, Z_ord_lower, Z_ord_upper) 
            self._W, self._sigma = W_new, sigma_new
            # stop early if the change in the correlation estimation is below the threshold
            loglik.append(iterloglik) 
            err = self._get_scaled_diff(prev_W, self._W)

            if err < self._threshold:
                break
            if len(loglik) > 1 and self._get_scaled_diff(loglik[-2], loglik[-1]) < 0.01:
                if self._verbose>0: 
                    print('early stop because changed likelihood below 1%')
                break
            if self._verbose>0:
                print(f'Interation {i+1}: noise ratio estimate {self._sigma:.3f}, copula parameter update ratio {err:.3f}, likelihood {iterloglik:.3f}')

        if self._verbose>0 and i == self._max_iter-1: 
            print("Convergence not achieved at maximum iterations")
        self.n_iter_ = i+1
        return Z, C


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
        """
        EM algorithm to estimate the low rank Gaussian copula, W and sigma.
        Args:
            Z (matrix): the transformed value, at observed continuous entry; 
                        initial conditional mean, at observed ordinal entry (will be updated during iteration); NA elsewhere
            r_lower, r_upper (matrix): the lower and upper bounds for con
        Returns:
            W (matrix): an estimate of the latent coefficient matrix of the low rank Gaussian copula
            sigma (scalar): an estimate of the latent noise variance of the low rank Gaussian copula
            C (matrix): 0 at observed continuous entry; the conditional variance, at observed ordinal entry; NA elsewhere
            loglik: log likelihood during iterations, expected to increase every iteration, but possible that it does not (indicating bad fit)

        """
        assert len(self._W.shape)==2, f'invalid W shape {self._W.shape}'
        n,p = Z.shape
        W, sigma = self._W, self._sigma
        rank = W.shape[1]
        if r_lower.shape[1] == 0:
            num_ord = 0
        else:
            num_ord = r_lower.shape[1]
        negloglik = 0
        U,d,V = np.linalg.svd(W, full_matrices=False)
        A = np.zeros((n, rank, rank))
        SS = np.copy(A)
        S = np.zeros((n,rank))
        C = np.zeros((n,p))

        # The main loop for the E step, parallelize this later
        for i in range(n):
            # indexing
            obs_indices = np.nonzero(~np.isnan(Z[i,:]))[0]
            ord_in_obs = np.nonzero(obs_indices < num_ord)
            ord_obs_indices = obs_indices[ord_in_obs]
            

            zi_obs = Z[i,obs_indices]
            Ui_obs = U[obs_indices,:]
            UU_obs = np.dot(Ui_obs.T, Ui_obs) 
            # YX: better edit to avoid vector-vector inner product when there is only one observation

            # used in both ordinal and factor block
            res = np.linalg.solve(UU_obs + sigma * np.diag(1.0/np.square(d)), np.concatenate((np.identity(rank), Ui_obs.T),axis=1))
            Ai = res[:,:rank]
            AU = res[:,rank:]
            A[i,:,:] = Ai

            # when there is an observed ordinal to be imputed and another observed dimension, impute this ordinal
            if len(obs_indices) >= 2 and len(ord_obs_indices) >= 1:
                #print("ENTERED INNER LOOP!!!")
                mu = (zi_obs - np.dot(Ui_obs, np.dot(AU, zi_obs)))/sigma
                for ind in range(len(obs_indices)):
                    j = obs_indices[ind]
                    if j < num_ord:
                        sigma_ij = sigma/(1 - np.dot(U[j,:].T, np.dot(Ai, U[j,:])))
                        mu_ij = Z[i,j] - mu[ind] * sigma_ij
                        mu_ij_new, sigma_ij_new = truncnorm.stats(
                            a=(r_lower[i,j] - mu_ij) / np.sqrt(sigma_ij),
                            b=(r_upper[i,j] - mu_ij) / np.sqrt(sigma_ij),
                            loc=mu_ij, scale=np.sqrt(sigma_ij),moments='mv')
                        if np.isfinite(sigma_ij_new):
                            C[i,j] = sigma_ij_new
                        else:
                            print("variance was not finite and is: " +str(sigma_ij_new))
                        if np.isfinite(mu_ij_new):
                            Z[i,j] = mu_ij_new
                        else:
                            print("mean was not finite and is: " +str(mu_ij_new))

            si = np.dot(AU, zi_obs)
            S[i,:] = si
            SS[i,:,:] = np.dot(AU * C[i, obs_indices], AU.T) + np.outer(si, si.T)
            negloglik = negloglik + np.log(sigma) * p + np.linalg.slogdet(np.identity(rank) + np.outer(d/sigma, d) * UU_obs)[1]
            negloglik = negloglik + np.sum(zi_obs**2) - np.dot(zi_obs.T, np.dot(Ui_obs, si))

        #print(negloglik)
        # M-step in W iterate over p
        W_new = np.copy(W)
        s = np.sum(C)

        for j in range(p):
            index_j = np.nonzero(~np.isnan(Z[:,j]))[0]
            # numerator
            rj = self._sum_2d_scale(M=S, c=Z[:,j], index=index_j) + np.dot(self._sum_3d_scale(A, c=C[:,j], index=index_j), U[j,:])
            # denominator
            Fj = self._sum_3d_scale(SS+sigma*A, c=np.ones(n), index = index_j) 
            W_new[j,:] = np.linalg.solve(Fj,rj) 
            s = s -  np.dot(rj, W_new[j,:])

        s1 = s
        #print('cross numerator: '+str(s1/float(np.sum(~np.isnan(Z)))))


        # M-step in sigma^2
        for i in range(n):
            obs_indices = np.nonzero(~np.isnan(Z[i,:]))
            zi_obs = Z[i,obs_indices]
            s += np.sum(zi_obs**2)
        #print('z numerator: '+str((s-s1)/float(np.sum(~np.isnan(Z)))))
        

        sigma_new = s/float(np.sum(~np.isnan(Z)))
        #print(sigma_new)
        W_new = np.dot(W_new * d, V)
        W, sigma = self._scale_corr(W_new, sigma_new)
        #print(sigma)
        loglik = -negloglik/2.0
        return W, sigma, Z, C, loglik/n



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



    def _sum_3d_scale(self, M, c, index):
        res = np.empty((M.shape[1], M.shape[2]))
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                #res[j,k] = np.sum(M[:,j,k][index] * c[index])
                res[j,k] = np.sum(M[index,j,k] * c[index])
        return res

    def _sum_2d_scale(self, M, c, index):
        res = np.empty(M.shape[1])
        for j in range(M.shape[1]):
            #res[j] = np.sum(M[:,j][index] * c[index])
            res[j] = np.sum(M[index,j] * c[index])
        return res

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






