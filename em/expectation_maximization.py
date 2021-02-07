from transforms.transform_function import TransformFunction
from scipy.stats import norm, truncnorm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from em.embody import _em_step_body_, _em_step_body, _em_step_body_row

class ExpectationMaximization():
    def __init__(self):
        return

    def impute_missing(self, X, cont_indices=None, ord_indices=None, threshold=0.01, max_iter=50, max_workers=4, max_ord=20, num_ord_updates=1, verbose=False, seed=1):
        """
        Fits a Gaussian Copula and imputes missing values in X.

        Args:
            X (matrix): data matrix with entries to be imputed
            cont_indices (array): logical, true at indices of the continuous entries
            ord_indices (array): logical, true at indices of the ordinal entries
            threshold (float): the threshold for scaled difference between covariance estimates at which to stop early
            max_iter (int): the maximum number of iterations for copula estimation
            max_workers: the maximum number of workers for parallelism
            max_ord: maximum number of levels in any ordinal for detection of ordinal indices
        Returns:
            X_imp (matrix): X with missing values imputed
            sigma_rearragned (matrix): an estimate of the covariance of the copula
        """
        if cont_indices is None and ord_indices is None:
            # guess the indices from the data
            cont_indices = self.get_cont_indices(X, max_ord)
            ord_indices = ~cont_indices
        self.transform_function = TransformFunction(X, cont_indices, ord_indices) ## estimate transformation function
        sigma, Z_imp = self._fit_covariance(X, cont_indices, ord_indices, threshold, max_iter, max_workers, num_ord_updates, verbose, seed)
        # rearrange sigma so it corresponds to the column ordering of X ## first few dims are always continuous, after always ordinal
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

    def _fit_covariance(self, X, cont_indices, ord_indices, threshold=0.01, max_iter=100, max_workers=4, num_ord_updates=1, verbose=False, seed=1):
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
            max_workers (positive int): the maximum number of workers for parallelism 

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
        # track previous sigma for the purpose of early stopping
        prev_sigma = self._project_to_correlation(sigma)
        for i in range(max_iter):
            sigma, Z_imp, Z = self._em_step(Z, Z_ord_lower, Z_ord_upper, sigma, max_workers, num_ord_updates)
            sigma = self._project_to_correlation(sigma)
            # stop early if the change in the correlation estimation is below the threshold
            sigmaudpate = self._get_scaled_diff(prev_sigma, sigma)
            if sigmaudpate < threshold:
                if verbose: print('Convergence at iteration '+str(i+1))
                break
            if verbose: print("Copula correlation change ratio: ", np.round(sigmaudpate, 4))
            prev_sigma = sigma
        if verbose and i == max_iter-1: 
            print("Convergence not achieved at maximum iterations")
        return sigma, Z_imp

    def _em_step(self, Z, r_lower, r_upper, sigma, max_workers=1, num_ord_updates=1):
        """
        Executes one step of the EM algorithm to update the covariance 
        of the copula

        Args:
            Z (matrix): Latent values
            r_lower (matrix): lower bound on latent ordinals
            r_upper (matrix): upper bound on latent ordinals
            sigma (matrix): correlation estimate
            max_workers (positive int): maximum number of workers for parallelism

        Returns:
            sigma (matrix): an estimate of the covariance of the copula
            Z_imp (matrix): estimates of latent values
            Z (matrix): Updated latent values

        """
        n = Z.shape[0]
        p = Z.shape[1]
        res = []
        if max_workers is None:
            max_workers = min(32, os.cpu_count()+4)
        divide = n/max_workers * np.arange(max_workers+1)
        divide = divide.astype(int)
        args = [(np.copy(Z[divide[i]:divide[i+1],:]), r_lower[divide[i]:divide[i+1],:], r_upper[divide[i]:divide[i+1],:], sigma, num_ord_updates) \
                               for i in range(max_workers)]
        with ProcessPoolExecutor(max_workers=max_workers) as pool: 
            res = pool.map(_em_step_body_, args)
            Z_imp = np.empty((n,p))
            C = np.zeros((p,p))
            for i,(C_divide, Z_imp_divide, Z_divide) in enumerate(res):
                C += C_divide/n
                Z_imp[divide[i]:divide[i+1],:] = Z_imp_divide
                Z[divide[i]:divide[i+1],:] = Z_divide
            sigma = np.cov(Z_imp, rowvar=False) + C 
            return sigma, Z_imp, Z

    def _project_to_correlation(self, covariance):
        """
        Projects a covariance to a correlation matrix, normalizing it's diagonal entries

        Args:
            covariance (matrix): a covariance matrix

        Returns:
            correlation (matrix): the covariance matrix projected to a correlation matrix
        """
        D = np.diagonal(covariance)
        D_neg_half = 1.0/np.sqrt(D)
        covariance *= D_neg_half
        return covariance.T * D_neg_half

    def _init_Z_ord(self, Z_ord_lower, Z_ord_upper, seed):
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

        np.random.seed(seed)
        for i in range(n):
            for j in range(k):
                if not np.isnan(Z_ord_upper[i,j]) and u_upper[i,j] > 0 and u_lower[i,j]<1:
                    u_sample = np.random.uniform(u_lower[i,j],u_upper[i,j])
                    Z_ord[i,j] = norm.ppf(u_sample)
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


