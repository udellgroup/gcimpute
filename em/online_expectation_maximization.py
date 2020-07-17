from transforms.online_transform_function import OnlineTransformFunction
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
            for j in ord_obs_indices:
                j_in_obs = np.where(obs_indices == j)[0]
                not_j_in_obs = np.where(obs_indices != j)[0]
                v = sigma_obs_obs_inv[:,j_in_obs]
                new_var_ij = np.asscalar(1.0/v[j_in_obs])
                new_mean_ij = np.asscalar(np.matmul(v[not_j_in_obs].T, Z_row[obs_indices[not_j_in_obs]])*(-new_var_ij))
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


class OnlineExpectationMaximization():
    def __init__(self, cont_indices, ord_indices):
        self.transform_function = OnlineTransformFunction(cont_indices, ord_indices)
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        # we assume boolean array of indices
        p = len(cont_indices)
        self.sigma = np.identity(p)
        # track what iteration the algorithm is on for use in weighting samples
        self.iteration = 1

    def partial_fit_and_predict(self, X_batch, max_workers=None, num_ord_updates=2, decay_coef=0.1):
        """
        Updates the fit of the copula using the data in X_batch and returns the 
        imputed values and the new correlation for the copula

        Args:
            X_batch (matrix): data matrix with entries to use to update copula and be imputed
            max_workers (positive int): the maximum number of workers for parallelism 
            num_ord_updates (positive int): the number of times to re-estimate the latent ordinals per batch
            decay_coef (float in (0,1)): tunes how much to weight new covariance estimates
        Returns:
            X_imp (matrix): X_batch with missing values imputed
            sigma_rearragned (matrix): an updated estimate of the covariance of the copula
        """
        # update marginals with the new batch
        self.transform_function.partial_fit(X_batch)
        sigma, Z_batch_imp = self._fit_covariance(X_batch, max_workers, num_ord_updates, decay_coef)
        # rearrange sigma so it corresponds to the column ordering of X
        sigma_rearranged = np.empty(sigma.shape)
        sigma_rearranged[np.ix_(self.ord_indices,self.ord_indices)] = sigma[:np.sum(self.ord_indices),:np.sum(self.ord_indices)]
        sigma_rearranged[np.ix_(self.cont_indices,self.cont_indices)] = sigma[np.sum(self.ord_indices):,np.sum(self.ord_indices):]
        sigma_rearranged[np.ix_(self.cont_indices,self.ord_indices)] = sigma[np.sum(self.ord_indices):,:np.sum(self.ord_indices)]
        sigma_rearranged[np.ix_(self.ord_indices,self.cont_indices)] =  sigma_rearranged[np.ix_(self.cont_indices,self.ord_indices)].T
        # Rearrange Z_imp so that it's columns correspond to the columns of X
        Z_imp_rearranged = np.empty(X_batch.shape)
        Z_imp_rearranged[:,self.ord_indices] = Z_batch_imp[:,:np.sum(self.ord_indices)]
        Z_imp_rearranged[:,self.cont_indices] = Z_batch_imp[:,np.sum(self.ord_indices):]
        X_imp_cont = np.copy(X_batch[:,self.cont_indices])
        X_imp_ord = np.copy(X_batch[:,self.ord_indices])
        # Impute continuous
        X_imp_cont[np.isnan(X_imp_cont)] = self.transform_function.partial_evaluate_cont_observed(Z_imp_rearranged)[np.isnan(X_imp_cont)]
        # Impute ordinal
        X_imp_ord[np.isnan(X_imp_ord)] = self.transform_function.partial_evaluate_ord_observed(Z_imp_rearranged)[np.isnan(X_imp_ord)]
        X_imp = np.empty(X_batch.shape)
        X_imp[:,self.cont_indices] = X_imp_cont
        X_imp[:,self.ord_indices] = X_imp_ord
        return X_imp, sigma_rearranged
    def _fit_covariance(self, X_batch, max_workers, num_ord_updates, decay_coef):
        """
        Updates the covariance matrix of the gaussian copula using the data 
        in X_batch and returns the imputed latent values corresponding to 
        entries of X_batch and the new sigma

        Args:
            X_batch (matrix): data matrix with which to update copula and with entries to be imputed
            max_workers: the maximum number of workers for parallelism 
            num_ord_updates: the number of times to restimate the latent ordinals per batch
            decay_coef (float in (0,1)): tunes how much to weight new covariance estimates
        Returns:
            sigma (matrix): an updated estimate of the covariance of the copula
            Z_imp (matrix): estimates of latent values in X_batch
        """
        Z_ord = None
        if self.ord_indices is not None:
           Z_ord_lower, Z_ord_upper = self.transform_function.partial_evaluate_ord_latent(X_batch)
           Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper)
        Z_cont = None
        if self.cont_indices is not None:
            Z_cont = self.transform_function.partial_evaluate_cont_latent(X_batch)
        Z_imp = np.concatenate((Z_ord,Z_cont), axis=1)
        # mean impute the missing continuous values for the sake of covariance estimation
        Z_imp[np.isnan(Z_imp)] = 0.0
        # Latent variable matrix with columns sorted as ordinal, continuous
        Z = np.concatenate((Z_ord, Z_cont), axis=1)
        batch_size = Z.shape[0]
        p = Z.shape[1]
        if np.all(np.isnan(Z_ord_lower)):
            num_ord = 0
        else:
            num_ord = Z_ord_lower.shape[1]
        # track previous sigma for the purpose of early stopping
        prev_sigma = self._project_to_correlation(self.sigma)
        Z_imp = np.zeros((batch_size, p))
        C = np.zeros((p, p))
        args = [(np.copy(Z[i,:]), np.copy(Z_ord_lower[i,:]), np.copy(Z_ord_upper[i,:]), prev_sigma, num_ord, num_ord_updates) for i in range(batch_size)]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            res = pool.map(_em_step_body_, args)
            for i,(C_row, Z_imp_row, Z_row) in enumerate(res):
                Z_imp[i,:] = Z_imp_row
                Z[i,:] = Z_row
                C += C_row/float(batch_size)
        sigma = np.cov(Z_imp, rowvar=False) + C
        sigma = self._project_to_correlation(sigma)
        self.sigma = sigma*decay_coef + (1 - decay_coef)*prev_sigma
        prev_sigma = self.sigma
        self.iteration += 1
        return self.sigma, Z_imp

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


