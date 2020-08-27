from transforms.transform_function import TransformFunction
from scipy.stats import norm, truncnorm
from em.expectation_maximization import ExpectationMaximization
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def _batch_em_step_body_(args):
    """
    Does a step of the EM algorithm, needed to dereference args to support parallelism
    """
    return _batch_em_step_body(*args)

def _batch_em_step_body(Z, r_lower, r_upper, sigma, num_ord_updates):
    """
    Iterate the rows over provided matrix 
    """
    num, p = Z.shape
    Z_imp = np.copy(Z)
    C = np.zeros((p,p))
    for i in range(num):
        c, z_imp, z = _batch_em_step_body_row(Z[i,:], r_lower[i,:], r_upper[i,:], sigma, num_ord_updates)
        Z_imp[i,:] = z_imp
        Z[i,:] = z
        C += c
    return C, Z_imp, Z

def _batch_em_step_body_row(Z_row, r_lower_row, r_upper_row, sigma, num_ord_updates):
    """
    The body of the em algorithm for each row
    Returns a new latent row, latent imputed row and C matrix, which, when added
    to the empirical covariance gives the expected covariance
    Args:
        Z_row (array): (potentially missing) latent entries for one data point
        r_lower_row (array): (potentially missing) lower range of ordinal entries for one data point
        r_upper_row (array): (potentially missing) upper range of ordinal entries for one data point
        sigma (matrix): estimate of covariance
        num_ord (int): the number of ordinal columns
        num_ord_updates (int): number of times to estimate ordinal mean
    Returns:
        C (matrix): results in the updated covariance when added to the empircal covariance
        Z_imp_row (array): Z_row with latent ordinals updated and missing entries imputed 
        Z_row (array): inpute Z_row with latent ordinals updated
    """
    Z_imp_row = np.copy(Z_row)
    p = Z_row.shape[0]
    num_ord = r_upper_row.shape[0]
    C = np.zeros((p,p))
    obs_indices = np.where(~np.isnan(Z_row))[0]
    missing_indices = np.setdiff1d(np.arange(p), obs_indices) ## Use set difference to avoid another searching
    ord_in_obs = np.where(obs_indices < num_ord)[0]
    ord_obs_indices = obs_indices[ord_in_obs]

    # obtain correlation sub-matrices
    # obtain submatrices by indexing a "cartesian-product" of index arrays
    sigma_obs_obs = sigma[np.ix_(obs_indices,obs_indices)]
    sigma_obs_missing = sigma[np.ix_(obs_indices, missing_indices)]
    sigma_missing_missing = sigma[np.ix_(missing_indices, missing_indices)]

    if len(missing_indices) > 0:
        tot_matrix = np.concatenate((np.identity(len(sigma_obs_obs)), sigma_obs_missing), axis=1)
        intermed_matrix = np.linalg.solve(sigma_obs_obs, tot_matrix)
        sigma_obs_obs_inv = intermed_matrix[:, :len(sigma_obs_obs)]
        J_obs_missing = intermed_matrix[:, len(sigma_obs_obs):]
    else:
        sigma_obs_obs_inv = np.linalg.solve(sigma_obs_obs, np.identity(len(sigma_obs_obs)))
    # initialize vector of variances for observed ordinal dimensions
    var_ordinal = np.zeros(p)

    # OBSERVED ORDINAL ELEMENTS
    # when there is an observed ordinal to be imputed and another observed dimension, impute this ordinal
    if len(obs_indices) >= 2 and len(ord_obs_indices) >= 1:
        for update_iter in range(num_ord_updates):
            # used to efficiently compute conditional mean
            sigma_obs_obs_inv_Z_row = np.dot(sigma_obs_obs_inv, Z_row[obs_indices])
            for ind in range(len(ord_obs_indices)):
                j = obs_indices[ind]
                not_j_in_obs = np.setdiff1d(np.arange(len(obs_indices)),ind) 
                v = sigma_obs_obs_inv[:,ind]
                new_var_ij = np.asscalar(1.0/v[ind])
                #new_mean_ij = np.dot(v[not_j_in_obs], Z_row[obs_indices[not_j_in_obs]]) * (-new_var_ij)
                new_mean_ij = Z_row[j] - new_var_ij*sigma_obs_obs_inv_Z_row[ind]
                mean, var = truncnorm.stats(
                    a=(r_lower_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                    b=(r_upper_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                    loc=new_mean_ij,
                    scale=np.sqrt(new_var_ij),
                    moments='mv')
                if np.isfinite(var):
                    var_ordinal[j] = var
                    if update_iter == num_ord_updates - 1:
                        C[j,j] = C[j,j] + var 
                if np.isfinite(mean):
                    Z_row[j] = mean

    Z_obs = Z_row[obs_indices]
    Z_imp_row[obs_indices] = Z_obs
    # MISSING ELEMENTS
    if len(missing_indices) > 0:
        Z_imp_row[missing_indices] = np.matmul(J_obs_missing.T,Z_obs)
        # variance expectation and imputation
        if len(ord_obs_indices) >= 1 and len(obs_indices) >= 2 and np.sum(var_ordinal) > 0:
            cov_missing_obs_ord = J_obs_missing[ord_in_obs].T * var_ordinal[ord_obs_indices]
            C[np.ix_(missing_indices, ord_obs_indices)] += cov_missing_obs_ord
            C[np.ix_(ord_obs_indices, missing_indices)] += cov_missing_obs_ord.T
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing) + np.matmul(cov_missing_obs_ord, J_obs_missing[ord_in_obs])
        else:
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing)
    return C, Z_imp_row, Z_row




class BatchExpectationMaximization(ExpectationMaximization):
    def impute_missing(self, X, cont_indices=None, ord_indices=None, threshold=0.01, max_iter=100, max_workers=1, max_ord=100, batch_size=64, num_ord_updates=2):
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
        sigma, Z_imp = self._fit_covariance(X, cont_indices, ord_indices, threshold, max_iter, max_workers, batch_size, num_ord_updates)
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

    def _fit_covariance(self, X, cont_indices, ord_indices, threshold, max_iter, max_workers, batch_size, num_ord_updates):
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
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper)
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
            divide = batch_size/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [(Z[indices[divide[i]:divide[i+1]],:], Z_ord_lower[indices[divide[i]:divide[i+1]],:], Z_ord_upper[indices[divide[i]:divide[i+1]],:], sigma, num_ord_updates) for i in range(max_workers)]
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_batch_em_step_body_, args)
                for i,(C_divide, Z_imp_divide, Z_divide) in enumerate(res):
                    Z_imp[indices[divide[i]:divide[i+1]],:] = Z_imp_divide
                    Z[indices[divide[i]:divide[i+1]],:] = Z_divide
                    C_batch += C_divide/batch_size
            sigma_batch = np.cov(Z_imp[indices,:], rowvar=False) + C_batch
            sigma_batch = self._project_to_correlation(sigma_batch)
            decay_coef = 1/(np.sqrt(batch_iter + 1))
            sigma = sigma_batch*decay_coef + (1 - decay_coef)*prev_sigma
            if self._get_scaled_diff(prev_sigma, sigma) < threshold:
                break
            prev_sigma = sigma
        return sigma, Z_imp

