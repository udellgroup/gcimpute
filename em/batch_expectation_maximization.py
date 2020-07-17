from transforms.transform_function import TransformFunction
from scipy.stats import norm, truncnorm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def _em_step_body_(args):
    """
    Does a step of the EM algorithm, needed to dereference args to support parallelism
    """
    return _em_step_body(*args)

def _em_step_body(Z_row, r_lower_row, r_upper_row, sigma, num_ord, num_ord_updates):
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
    p = Z_row.shape[0]
    Z_imp_row = np.copy(Z_row)
    C = np.zeros((p,p))
    obs_indices = np.where(~np.isnan(Z_row))[0]
    missing_indices = np.where(np.isnan(Z_row))[0]
    ord_in_obs = np.where(obs_indices < num_ord)[0]
    ord_obs_indices = obs_indices[ord_in_obs]
    # obtain correlation sub-matrices
    # obtain submatrices by indexing a "cartesian-product" of index arrays
    sigma_obs_obs = sigma[np.ix_(obs_indices,obs_indices)]
    sigma_obs_missing = sigma[np.ix_(obs_indices, missing_indices)]
    sigma_missing_missing = sigma[np.ix_(missing_indices, missing_indices)]
    # precompute psuedo-inverse 
    sigma_obs_obs_inv = np.linalg.pinv(sigma_obs_obs)
    # precompute sigma_obs_obs_inv * simga_obs_missing
    if len(missing_indices) > 0:
        J_obs_missing = np.matmul(sigma_obs_obs_inv, sigma_obs_missing)
    # initialize vector of variances for observed ordinal dimensions
    var_ordinal = np.zeros(p)

    # OBSERVED ORDINAL ELEMENTS
    # when there is an observed ordinal to be imputed and another observed dimension, impute this ordinal
    if len(obs_indices) >= 2 and len(ord_obs_indices) >= 1:
        for update_iter in range(num_ord_updates):
            # used to efficiently compute conditional mean
            sigma_obs_obs_inv_Z_row = np.matmul(sigma_obs_obs_inv, Z_row[obs_indices])
            for j in ord_obs_indices:
                j_in_obs = np.where(obs_indices == j)[0]
                not_j_in_obs = np.where(obs_indices != j)[0]
                v = sigma_obs_obs_inv[:,j_in_obs]
                new_var_ij = np.asscalar(1.0/v[j_in_obs])
                new_mean_ij = Z_row[j] - new_var_ij*sigma_obs_obs_inv_Z_row[j_in_obs]
                # the boundaries must be de-meaned and normalized
                mean, var = truncnorm.stats(
                    a=(r_lower_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                    b=(r_upper_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                    loc=new_mean_ij,
                    scale=np.sqrt(new_var_ij),
                    moments='mv'
                )
                if np.isfinite(var):
                    var_ordinal[j] = var
                    if update_iter == num_ord_updates - 1:
                        # update the variance estimate
                        C[j,j] = C[j,j] + var
                if np.isfinite(mean):
                    Z_row[j] = mean
    Z_obs = Z_row[obs_indices]
    # mean expection and imputation
    Z_imp_row[obs_indices] = Z_obs
    # MISSING ELEMENTS
    if len(missing_indices) > 0:
        Z_imp_row[missing_indices] = np.matmul(J_obs_missing.T,Z_obs)
        # variance expectation and imputation
        if len(ord_obs_indices) >= 1 and len(obs_indices) >= 2 and np.sum(var_ordinal) > 0:
            diag_var_ord = np.diag(var_ordinal[ord_obs_indices])
            cov_missing_obs_ord = np.matmul(J_obs_missing[ord_in_obs].T, diag_var_ord)
            C[np.ix_(missing_indices, ord_obs_indices)] += cov_missing_obs_ord
            C[np.ix_(ord_obs_indices, missing_indices)] += cov_missing_obs_ord.T
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing) + np.matmul(cov_missing_obs_ord, J_obs_missing[ord_in_obs])
        else:
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing)
    return C, Z_imp_row, Z_row

class BatchExpectationMaximization():
    def impute_missing(self, X, cont_indices=None, ord_indices=None, threshold=0.01, max_iter=100, max_workers=None, max_ord=100, batch_size=64, num_ord_updates=2):
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
        X_imp_cont = np.copy(X[:,cont_indices])
        X_imp_ord = np.copy(X[:,ord_indices])
        # Impute continuous
        X_imp_cont[np.isnan(X_imp_cont)] = self.transform_function.impute_cont_observed(Z_imp_rearranged)[np.isnan(X_imp_cont)]
        # Impute ordinal
        X_imp_ord[np.isnan(X_imp_ord)] = self.transform_function.impute_ord_observed(Z_imp_rearranged)[np.isnan(X_imp_ord)]
        X_imp = np.empty(X.shape)
        X_imp[:,cont_indices] = X_imp_cont
        X_imp[:,ord_indices] = X_imp_ord
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
        assert cont_indices is not None or ord_indices is not None
        Z_ord = None
        if ord_indices is not None:
           Z_ord_lower, Z_ord_upper = self.transform_function.get_ord_latent()
           Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper)
        Z_cont = None
        if cont_indices is not None:
            Z_cont = self.transform_function.get_cont_latent()
        Z_imp = np.concatenate((Z_ord,Z_cont), axis=1)
        # mean impute the missing continuous values for the sake of covariance estimation
        Z_imp[np.isnan(Z_imp)] = 0.0
        # initialize the correlation matrix
        sigma = np.corrcoef(Z_imp, rowvar=False)
        # Latent variable matrix with columns sorted as ordinal, continuous
        Z = np.concatenate((Z_ord, Z_cont), axis=1)
        n = Z.shape[0]
        p = Z.shape[1]
        if np.all(np.isnan(Z_ord_lower)):
            num_ord = 0
        else:
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
            args = [(np.copy(Z[i,:]), np.copy(Z_ord_lower[i,:]), np.copy(Z_ord_upper[i,:]), sigma, num_ord, num_ord_updates) for i in indices]
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_em_step_body_, args)
                for i,(C_row, Z_imp_row, Z_row) in enumerate(res):
                    Z_imp[indices[i],:] = Z_imp_row
                    Z[indices[i],:] = Z_row
                    C_batch += C_row/batch_size
            sigma_batch = np.cov(Z_imp[indices,:], rowvar=False) + C_batch
            sigma_batch = self._project_to_correlation(sigma_batch)
            decay_coef = 1/(np.sqrt(batch_iter + 1))
            sigma = sigma_batch*decay_coef + (1 - decay_coef)*prev_sigma
            if self._get_scaled_diff(prev_sigma, sigma) < threshold:
                break
            prev_sigma = sigma
        return sigma, Z_imp

    def _project_to_correlation(self, covariance):
        """
        Projects a covariance to a correlation matrix, normalizing it's diagonal entries
        Args:
            covariance (matrix): a covariance matrix
        Returns:
            correlation (matrix): the covariance matrix projected to a correlation matrix
        """
        D = np.diagonal(covariance)
        D_neg_half = np.diag(1.0/np.sqrt(D))
        return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

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
        u_lower = np.copy(Z_ord_lower)
        u_lower[~np.isnan(Z_ord_lower)] = norm.cdf(Z_ord_lower[~np.isnan(Z_ord_lower)])
        u_upper = np.copy(Z_ord_upper)
        u_upper[~np.isnan(Z_ord_upper)] = norm.cdf(Z_ord_upper[~np.isnan(Z_ord_upper)])
        u_samples = np.random.uniform(u_lower[~np.isnan(u_lower)],u_upper[~np.isnan(u_lower)])
        # convert back from the uniform sample to the guassian sample in that interval
        Z_ord[~np.isnan(u_lower)] = norm.ppf(u_samples)
        return Z_ord

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

    def get_cont_indices(self, X, max_ord):
        """
        get's the indices of continuos columns by returning
        those indicies which have at least max_ord distinct values
        Args:
            X (matrix): input matrix
            max_ord (int): maximum number of distinct values an ordinal can take on in a column
        Returns:
            indices (array): indices of the columns which have at most max_ord distinct entries
        """
        indices = np.zeros(X.shape[1]).astype(bool)
        for i, col in enumerate(X.T):
            col_nonan = col[~np.isnan(col)]
            col_unique = np.unique(col_nonan)
            if len(col_unique) > max_ord:
                indices[i] = True
        return indices