from .transform_function import TransformFunction
from .gaussian_copula import GaussianCopula
from scipy.stats import norm, truncnorm
import numpy as np


class LowRankGaussianCopula(GaussianCopula):
    def __init__(self, var_types=None, max_ord=20):
        if var_types is not None:
            if not all(var_types['cont'] ^ var_types['ord']):
                raise ValueError('Inconcistent specification of variable types indexing')
            self.cont_indices = var_types['cont']
            self.ord_indices = var_types['ord']
        else:
            self.cont_indices = None
            self.ord_indices = None
        self.max_ord = max_ord


    def impute_missing(self, X, rank, threshold=1e-3, max_iter=50, max_ord=20, verbose = False, seed=1):
        """
        Fits a low rank Gaussian Copula and imputes missing values in X. After estimating the model parameters W and sigma, 
        a further step to update S (detemined by W, sigma, Z) is implemented for numerical stability
        Args:
            X (matrix): data matrix with entries to be imputed
            rank: the rank for low rank Gaussian copula 
            cont_indices (array): indices of the continuous entries
            ord_indices (array): indices of the ordinal entries
            threshold (float): the threshold for scaled difference between covariance estimates at which to stop early
            max_iter (int): the maximum number of iterations for copula estimation
            max_ord: maximum number of levels in any ordinal for detection of ordinal indices
            verbose: print iteration information if true
        Returns:
            X_imp (matrix): X with missing values imputed
            W (matrix): an estimate of the latent coefficient matrix of the low rank Gaussian copula
            sigma (scalar): an estimate of the latent noise variance of the low rank Gaussian copula
        """
        if self.cont_indices is None:
            self.cont_indices = self.get_cont_indices(X, self.max_ord)
            self.ord_indices = ~self.cont_indices

        self.transform_function = TransformFunction(X, self.cont_indices, self.ord_indices)
        # TO DO: consider the order of W
        W, sigma, Z, C, loglik = self._fit_covariance(X=X, rank=rank, threshold=threshold, max_iter=max_iter, verbose=verbose, seed=seed)
        S = self._comp_S(Z, W, sigma) # re-estimate S to ensure numerical stability
        Z_imp = self._impute(Z, S, W)
        # Rearrange Z_imp so that it's columns correspond to the columns of X
        #Z_imp_rearranged = np.empty(X.shape)
        #Z_imp_rearranged[:,ord_indices] = Z_imp[:,:np.sum(ord_indices)]
        #Z_imp_rearranged[:,cont_indices] = Z_imp[:,np.sum(ord_indices):]
        _order = self.back_to_original_order()
        Z_imp_rearranged = Z_imp[:,_order]

        X_imp = np.empty(X.shape)
        if np.sum(self.cont_indices) > 0:
            #X_imp[:,cont_indices] = self.transform_function.impute_cont_observed(Z_imp_rearranged)
            X_imp[:,self.cont_indices] = self.transform_function.impute_cont_observed(Z_imp_rearranged)
        if np.sum(self.ord_indices) >0:
            #X_imp[:,ord_indices] = self.transform_function.impute_ord_observed(Z_imp_rearranged)
            X_imp[:,self.ord_indices] = self.transform_function.impute_ord_observed(Z_imp_rearranged)

        return X_imp, W, sigma

    def _fit_covariance(self, X, rank, threshold=1e-3, max_iter =100, verbose = False, seed=1):
        """
        Estimate the covariance parameters of the low rank Gaussian copula, W and sigma, 
        using the data in X and return the estimates and related quantity. 
        Different from the full rank case, imputation is not conducted in this step.

        Args:
            X (matrix): data matrix with entries to be imputed
            rank: the rank for low rank Gaussian copula 
            threshold (float): the threshold for scaled difference between covariance estimates at which to stop early
            max_iter (int): the maximum number of iterations for copula estimation
            verbose: print iteration information if true
        Returns:
            W (matrix): an estimate of the latent coefficient matrix of the low rank Gaussian copula
            sigma (scalar): an estimate of the latent noise variance of the low rank Gaussian copula
            Z (matrix): the transformed value, at observed continuous entry; the conditional mean, at observed ordinal entry; NA elsewhere
            C (matrix): 0 at observed continuous entry; the conditional variance, at observed ordinal entry; NA elsewhere
            loglik: log likelihood during iterations, expected to increase every iteration, but possible that it does not (indicating bad fit)
        """
        Z_ord_lower, Z_ord_upper = self.transform_function.get_ord_latent()
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper, seed)
        Z_cont = self.transform_function.get_cont_latent()
        Z = np.concatenate((Z_ord, Z_cont), axis=1)

        # Initialize Z_imp using truncated (low-rank) SVD for missing entries
        # to obtain initial parameter estimate
        Z_imp = self._init_impute_svd(Z, rank, Z_ord_lower, Z_ord_upper)
        corr = np.corrcoef(Z_imp, rowvar=False)
        u,d,_ = np.linalg.svd(corr, full_matrices=False)
        sigma = np.mean(d[rank:])
        W = u[:,:rank] * (np.sqrt(d[:rank] - sigma))
        W, sigma = self._scale_corr(W, sigma)
        # Update entries at obseved ordinal locations from SVD initialization
        if sum(self.ord_indices)>0:
            Z_ord[~np.isnan(Z_ord)] = Z_imp[:,:sum(self.ord_indices)][~np.isnan(Z_ord)]
            Z = np.concatenate((Z_ord, Z_cont), axis=1)
        

        loglik = []
        for i in range(max_iter):
            #print("iteration " + str(i + 1))
            W_new, sigma_new, C, iterloglik = self._em_step(Z, Z_ord_lower, Z_ord_upper, W, sigma) # YX
            # stop early if the change in the correlation estimation is below the threshold
            #loglik.append(-negloglik)
            loglik.append(iterloglik) #YX
            err = self._get_scaled_diff(W, W_new)
            if err < threshold:
                return W_new, sigma_new, Z, C, loglik
            if len(loglik) > 1 and self._get_scaled_diff(loglik[-2], loglik[-1]) < 0.01:
                if verbose: print('early stop because changed likelihood below 1%')
                return W_new, sigma_new, Z, C, loglik
            if verbose:
                print('sigma estimate: '+ str(sigma))
                print('log likelihood: '+str(iterloglik))
                print('Updated error: '+str(err))
            sigma, W = sigma_new, W_new
        return W, sigma, Z, C, loglik


    def _comp_S(self, Z, W, sigma):
        """
        Intermidiate step.
        It seems that S must be updated using the last obtained W and sigma, otherwise numerical instability happens. 
        Such problem is not seen in R implementation. Keep an eye.
        Args:
            W (matrix): an estimate of the latent coefficient matrix of the low rank Gaussian copula
            sigma (scalar): an estimate of the latent noise variance of the low rank Gaussian copula
            Z (matrix): the transformed value, at observed continuous entry; the conditional mean, at observed ordinal entry; NA elsewhere
        Returns:
            S: a factor used for imputation
        """
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




    def _em_step(self, Z, r_lower, r_upper, W, sigma):
        """
        EM algorithm to estimate the low rank Gaussian copula, W and sigma.
        Args:
            Z (matrix): the transformed value, at observed continuous entry; 
                        initial conditional mean, at observed ordinal entry (will be updated during iteration); NA elsewhere
            r_lower, r_upper (matrix): the lower and upper bounds for con
            W, sigma: initial estimate for low rank Gaussian copula parameters
        Returns:
            W (matrix): an estimate of the latent coefficient matrix of the low rank Gaussian copula
            sigma (scalar): an estimate of the latent noise variance of the low rank Gaussian copula
            C (matrix): 0 at observed continuous entry; the conditional variance, at observed ordinal entry; NA elsewhere
            loglik: log likelihood during iterations, expected to increase every iteration, but possible that it does not (indicating bad fit)

        """
        n,p = Z.shape
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
            negloglik = negloglik + np.log(sigma) * p + np.log(np.linalg.det(np.identity(rank) + np.outer(d/sigma, d) * UU_obs))
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
        return W, sigma, C, loglik



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






