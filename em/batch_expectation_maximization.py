from transforms.transform_function import TransformFunction
from scipy.stats import norm, truncnorm
from em.expectation_maximization import ExpectationMaximization
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from em.embody import _em_step_body_, _em_step_body, _em_step_body_row


class BatchExpectationMaximization(ExpectationMaximization):
    def impute_missing(self, X, cont_indices=None, ord_indices=None, threshold=0.01, max_iter=100, max_workers=4, max_ord=100, batch_size=64, batch_c=1, num_ord_updates=1, verbose=False, seed=1):
        """
        Fits a Gaussian Copula and imputes missing values in X.
        Args:
            X (matrix): data matrix with entries to be imputed
            cont_indices (array): indices of the continuous entries
            ord_indices (array): indices of the ordinal entries
            threshold (float): the threshold for scaled difference between covariance estimates at which to stop early
            max_iter (int): the maximum number of iterations for copula estimation
            max_workers: the maximum number of workers for parallelism 
            max_ord: maximum number of levels in any ordinal for detection of ordinal indices
            batch_size: the number of elements to process in each iteration for copula update
            num_ord_updates: the number of times to re-estimate the latent ordinals per batch
        Returns:
            X_imp (matrix): X with missing values imputed
            sigma_rearragned (matrix): an estimate of the covariance of the copula
        """
        if cont_indices is None and ord_indices is None:
            # guess the indices from the data
            cont_indices = self.get_cont_indices(X, max_ord=max_ord)
            ord_indices = ~cont_indices
        self.transform_function = TransformFunction(X, cont_indices, ord_indices)
        sigma, Z_imp = self._fit_covariance(X, cont_indices, ord_indices, threshold, max_iter, max_workers, batch_size, batch_c, num_ord_updates, verbose, seed)
        # rearrange sigma so it corresponds to the column ordering of X
        sigma_rearranged = np.empty(sigma.shape)
        sigma_rearranged[np.ix_(ord_indices,ord_indices)] = sigma[:np.sum(ord_indices),:np.sum(ord_indices)]
        sigma_rearranged[np.ix_(cont_indices,cont_indices)] = sigma[np.sum(ord_indices):,np.sum(ord_indices):]
        sigma_rearranged[np.ix_(cont_indices,ord_indices)] = sigma[np.sum(ord_indices):,:np.sum(ord_indices)]
        sigma_rearranged[np.ix_(ord_indices,cont_indices)] =  sigma_rearranged[np.ix_(cont_indices,ord_indices)].T
        # Rearrange Z_imp so that it's columns correspond to the columns of X
        Z_imp_rearranged = np.empty(X.shape)
        Z_imp_rearranged[:,ord_indices] = Z_imp[:,:np.sum(ord_indices)]
        Z_imp_rearranged[:,cont_indices] = Z_imp[:,np.sum(ord_indices):]
        X_imp = np.empty(X.shape)
        X_imp[:,cont_indices] = self.transform_function.impute_cont_observed(Z_imp_rearranged)
        X_imp[:,ord_indices] = self.transform_function.impute_ord_observed(Z_imp_rearranged)
        return X_imp, sigma_rearranged

    def _fit_covariance(self, X, cont_indices, ord_indices, threshold, max_iter, max_workers=4, batch_size=64, batch_c=1, num_ord_updates=1, verbose=False, seed=1):
        """
        Fits the covariance matrix of the gaussian copula using the data 
        in X and returns the imputed latent values corresponding to 
        entries of X and the covariance of the copula
        Args:
            X (matrix): data matrix with entries to be imputed
            cont_indices (array): indices of the continuous entries
            ord_indices (array): indices of the ordinal entries
            threshold (float): the threshold for scaled difference between covariance estimates at which to stop early
            max_iter (int): the maximum number of iterations for copula estimation
            max_workers: the maximum number of workers for parallelism 
            batch_size: the number of elements to process in each iteration for copula update
            num_ord_updates: the number of times to restimate the latent ordinals per batch
        Returns:
            sigma (matrix): an estimate of the covariance of the copula
            Z_imp (matrix): estimates of latent values
        """
        assert cont_indices is not None or ord_indices is not None
        Z_ord_lower, Z_ord_upper = self.transform_function.get_ord_latent()
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper, seed)
        Z_cont = self.transform_function.get_cont_latent()
        Z_imp = np.concatenate((Z_ord,Z_cont), axis=1)
        # mean impute the missing continuous values for the sake of covariance estimation
        Z_imp[np.isnan(Z_imp)] = 0.0
        # initialize the correlation matrix
        sigma = np.corrcoef(Z_imp, rowvar=False)
        # Latent variable matrix with columns sorted as ordinal, continuous
        Z = np.concatenate((Z_ord, Z_cont), axis=1)
        n,p = Z.shape
        num_ord = Z_ord_lower.shape[1]
        # track previous sigma for the purpose of early stopping
        prev_sigma = self._project_to_correlation(sigma)
        # permutation of indices of data for stochastic fitting
        training_permutation = np.random.permutation(n)
        Z_imp = np.zeros((n, p))
        for batch_iter in range(max_iter):
            batch_lower = (batch_iter * batch_size) % n
            batch_upper = ((batch_iter+1) * batch_size) % n
            if batch_upper < batch_lower:
                # we have wrapped around the dataset
                indices = np.concatenate((training_permutation[batch_lower:], training_permutation[:batch_upper]))
            else:
                indices = training_permutation[batch_lower:batch_upper]
            C_batch = np.zeros((p, p))
            if max_workers is None:
                max_workers = min(32, os.cpu_count()+4)
            divide = batch_size/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [(Z[indices[divide[i]:divide[i+1]],:], Z_ord_lower[indices[divide[i]:divide[i+1]],:], Z_ord_upper[indices[divide[i]:divide[i+1]],:], sigma, num_ord_updates) for i in range(max_workers)]
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_em_step_body_, args)
                for i,(C_divide, Z_imp_divide, Z_divide) in enumerate(res):
                    Z_imp[indices[divide[i]:divide[i+1]],:] = Z_imp_divide
                    Z[indices[divide[i]:divide[i+1]],:] = Z_divide
                    C_batch += C_divide/batch_size
            sigma_batch = np.cov(Z_imp[indices,:], rowvar=False) + C_batch
            sigma_batch = self._project_to_correlation(sigma_batch)
            decay_coef = batch_c/(batch_iter + 1 + batch_c)
            sigma = sigma_batch*decay_coef + (1 - decay_coef)*prev_sigma
            if self._get_scaled_diff(prev_sigma, sigma) < threshold: 
            # in this situation, possibly update the imputation for all points that haven't been met during this pass
                if verbose: print('Convergence at batch iteration '+str(batch_iter+1))
                break
            prev_sigma = sigma
        if verbose and batch_iter == max_iter-1: 
            print("Convergence not achieved at maximum iterations")
        return sigma, Z_imp

