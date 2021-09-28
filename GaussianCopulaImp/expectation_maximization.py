from .transform_function import TransformFunction
from .online_transform_function import OnlineTransformFunction
from .embody import _em_step_body_, _em_step_body, _em_step_body_row
from scipy.stats import norm, truncnorm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import warnings
from scipy.linalg import svdvals
from collections import defaultdict

class ExpectationMaximization():
    '''
    A method to fit a Gaussian copul model from incomplete data and then use the fitted model to impute the missing entries.

    Attributes
    ----------
    cont_indices: list 
        list of Boolean variables of length p (the number of variables), indicating locations of continuous variables.
    ord_indices: list
        list of Boolean variables of length p, indicating locations of ordinal variables (including binary variables). 
    max_ord: int
        int. When cont_indices and ord_indices are not specified, variables whose numbers of unique values are regarded as continuous variables and others as ordinal variables.
    sigma: numpy array
        numpy array of shape (p, p), the copula correlation matrix. 

    Methods
    -------
    impute_missing:
        fit a Gaussian copula model from incomplete data and then use the fitted model to impute the missing entries.
    impute_missing_online:
        At each sequentially observed data batch, fit a Gaussian copula model from incomplete data and then use the fitted model to impute the missing entries.
    '''

    def __init__(self, var_types=None, max_ord=20, sigma_init = None):
        '''
        The user can tell the model which variables are continuous and which are ordinal by the following two ways:
        (1) input a dict var_types that contains valid assignmetns of cont_indices and ord_indices;
        (2) input a max_ord so that the variables whose number of unique observation below max_ord will be treated as ordinal variables.
        If both are provided, only var_types will be used.

        Args:
            var_types: dict
            max_ord: int
            sigma_init: None or numpy array
        '''
        if var_types is not None:
            if not all(var_types['cont'] ^ var_types['ord']):
                raise ValueError('Inconcistent specification of variable types indexing')
            self.cont_indices = var_types['cont']
            self.ord_indices = var_types['ord']
        else:
            self.cont_indices = None
            self.ord_indices = None 
        self.max_ord = max_ord
        if sigma_init is not None:
            message = 'the intial correlation matrix must be nonsingular, while the input has the smallest singular value below 1e-7'
            assert svdvals(sigma_init).min() > 1e-7, message
        self.sigma = sigma_init

    def impute_missing(self, X, threshold=0.01, max_iter=50, max_workers=1, num_ord_updates=1, 
                       batch_size=100, batch_c=0, 
                       window_size=200, const_decay = -1, 
                       verbose=False, seed=1):
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
        if self.cont_indices is None:
            self.cont_indices = self.get_cont_indices(X, self.max_ord)
            self.ord_indices = ~self.cont_indices

        #self._fit_initial_transformation(X, window_size)
        self.transform_function = TransformFunction(X, self.cont_indices, self.ord_indices)
        Z_imp = self._fit_covariance(X, threshold, max_iter, max_workers, num_ord_updates, batch_size, batch_c, verbose, seed)
        # rearrange sigma so it corresponds to the column ordering of X ## first few dims are always continuous, after always ordinal
        _order = self.back_to_original_order()
        # Rearrange Z_imp so that it's columns correspond to the columns of X
        Z_imp_rearranged = Z_imp[:,_order]
        X_imp = np.empty(X.shape)
        if np.sum(self.cont_indices) > 0:
            X_imp[:,self.cont_indices] = self.transform_function.impute_cont_observed(Z_imp_rearranged)
        if np.sum(self.ord_indices) >0:
            X_imp[:,self.ord_indices] = self.transform_function.impute_ord_observed(Z_imp_rearranged)
        sigma_rearranged = self.sigma[np.ix_(_order, _order)]

        return {'imputed_data':X_imp, 'copula_corr':sigma_rearranged}


    def _fit_covariance(self, X, 
                        threshold=0.01, max_iter=100, max_workers=4, num_ord_updates=1, 
                        batch_size=100, batch_c=0, 
                        verbose=False, seed=1):
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
        n,p = X.shape
        Z_ord_lower, Z_ord_upper = self.transform_function.get_ord_latent()
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper, seed)
        Z_cont = self.transform_function.get_cont_latent()

        Z_imp = np.concatenate((Z_ord,Z_cont), axis=1)
        # mean impute the missing continuous values for the sake of covariance estimation
        Z_imp[np.isnan(Z_imp)] = 0.0
        # initialize the correlation matrix
        sigma = np.corrcoef(Z_imp, rowvar=False)
        if self.sigma is None:
            self.sigma = np.corrcoef(Z_imp, rowvar=False)
        # Latent variable matrix with columns sorted as ordinal, continuous
        Z = np.concatenate((Z_ord, Z_cont), axis=1)
            

        # permutation of indices of data for stochastic fitting
        training_permutation = np.random.permutation(n)
        for i in range(max_iter):
            # track previous sigma for the purpose of early stopping
            prev_sigma = self.sigma
            if np.isnan(prev_sigma).any():
                raise ValueError(f'Unexpected nan in updated sigma at iteration {i}')

            # mini-batch EM: more frequent parameter update by using data input with smaller size at each iteration
            if batch_c>0:
                batch_lower = (i * batch_size) % n
                batch_upper = ((i+1) * batch_size) % n
                if batch_upper < batch_lower:
                    # we have wrapped around the dataset
                    indices = np.concatenate((training_permutation[batch_lower:], training_permutation[:batch_upper]))
                else:
                    indices = training_permutation[batch_lower:batch_upper]
                sigma, Z_imp_batch, Z_batch = self._em_step(Z[indices], Z_ord_lower[indices], Z_ord_upper[indices], max_workers, num_ord_updates)
                Z_imp[indices] = Z_imp_batch
                Z[indices] = Z_batch
                decay_coef = batch_c/(i + 1 + batch_c)
                self.sigma = sigma*decay_coef + (1 - decay_coef)*prev_sigma
            # standard EM: each iteration uses all data points
            else:
                sigma, Z_imp, Z = self._em_step(Z, Z_ord_lower, Z_ord_upper, max_workers, num_ord_updates)
                #print(f"at iteration {i}, sigma has {np.isnan(sigma).sum()} nan entries, Z_imp has {np.isnan(Z_imp).sum()} nan entries")
                self.sigma = sigma
            # stop early if the change in the correlation estimation is below the threshold
            sigmaudpate = self._get_scaled_diff(prev_sigma, self.sigma)
            if sigmaudpate < threshold:
                if verbose: 
                    print('Convergence at iteration '+str(i+1))
                break
            if verbose: 
                print("Copula correlation change ratio: ", np.round(sigmaudpate, 4))
            
        if verbose and i == max_iter-1: 
            print("Convergence not achieved at maximum iterations")
        return  Z_imp
    
    def impute_missing_online(self, X, 
                              threshold=0.01, max_workers=1, num_ord_updates=1, 
                              batch_size=100, batch_c=0, window_size=200, const_decay = -1, 
                              verbose=False, seed=1, sigma_diff=['F']):
        """
        Fit the Gaussian copula model at each new batch of data points. If the provided X is not an iterable but a numpy array, 
        an iterable will be constructed by sequentially iterating over X using the specified batch size. To take mutiple passes 
        through the data, input the stacked data (by the number of passes) as X. However, it is recommended to use offline batch 
        mode for that purpose.  
        Args:s
            X (matrix): data matrix with entries to be imputed
            cont_indices (array): logical, true at indices of the continuous entries
            ord_indices (array): logical, true at indices of the ordinal entries
            threshold (float): the threshold for scaled difference between covariance estimates at which to stop early
            max_iter (int): the maximum number of iterations for copula estimation
            max_workers: the maximum number of workers for parallelism
            max_ord: maximum number of levels in any ordinal for detection of ordinal indices
            sigma_diff: A subset of ['F', 'S', 'N']. 'F' for Frobenius norm, 'S' for spectral norm and 'N' for nuclear norm. 
        Returns:
            X_imp (matrix): X with missing values imputed
            sigma_rearragned (matrix): an estimate of the covariance of the copula
        """
        assert self.cont_indices is not None and self.ord_indices is not None, 'Variable types must be provided for online fit'
        self.transform_function = OnlineTransformFunction(self.cont_indices, self.ord_indices, window_size=window_size)
        n,p = X.shape
        X_imp = np.zeros_like(X)
        if self.sigma is None:
            self.sigma = np.identity(p)
        sigma_diff_output = defaultdict(list)

        i=0
        while True:
            batch_lower= i*batch_size
            batch_upper=min((i+1)*batch_size, n)
            if batch_lower>= n:
                break 
            indices = np.arange(batch_lower, batch_upper, 1)
            decay_coef = const_decay if 0<const_decay<1 else batch_c/(i + 1 + batch_c)
            out = self.partial_fit_and_predict(X[indices,:], max_workers=max_workers, decay_coef=decay_coef, num_ord_updates=num_ord_updates, sigma_diff=sigma_diff)
            X_imp[indices,:] = out['imputed']
            for k,v in out['sigma_diff'].items():
                sigma_diff_output[k].append(v)

            i+=1
        _order = self.back_to_original_order()
        sigma_rearranged = self.sigma[np.ix_(_order, _order)]
        return {'imputed_data':X_imp, 'copula_corr':sigma_rearranged, 'copula_corr_change':sigma_diff_output}

    # TO DO: add a function attribute which takes estimated model and new point as input to return immediate imputaiton
    #  that would serve as out-of-sample prediction without updating the model parameter. Computation will be smaller but the complexity is still O(p^3)
    def partial_fit_and_predict(self, X_batch, max_workers=4, num_ord_updates=2, decay_coef=0.5, sigma_update=True, marginal_update = True, seed = 1, sigma_diff=None):
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
        """
        
        #if not update:
            #old_window = self.transform_function.window
            #old_update_pos = self.transform_function.update_pos
        if marginal_update:
            self.transform_function.partial_fit(X_batch)
        # update marginals with the new batch
        #self.transform_function.partial_fit(X_batch)
        # print("X_batch", X_batch)
        #
        # _fit_covariance step
        Z_ord_lower, Z_ord_upper = self.transform_function.partial_evaluate_ord_latent(X_batch)
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper, seed)
        Z_cont = self.transform_function.partial_evaluate_cont_latent(X_batch) 
        Z = np.concatenate((Z_ord, Z_cont), axis=1)
        sigma, Z_imp, Z = self._em_step(Z, Z_ord_lower, Z_ord_upper, max_workers, num_ord_updates)
        prev_sigma = self.sigma
        if sigma_update:
            self.sigma = sigma*decay_coef + (1-decay_coef)*self.sigma

        diff = None if sigma_diff is None else self.get_matrix_diff(prev_sigma, self.sigma, sigma_diff)

        # rearrange sigma so it corresponds to the column ordering of X ## first few dims are always continuous, after always ordinal
        _order = self.back_to_original_order()
        Z_imp_rearranged = Z_imp[:,_order]
        X_imp = np.empty(Z_imp.shape)
        X_imp[:,self.cont_indices] = self.transform_function.partial_evaluate_cont_observed(Z_imp_rearranged, X_batch)
        X_imp[:,self.ord_indices] = self.transform_function.partial_evaluate_ord_observed(Z_imp_rearranged, X_batch)
        return {'imputed':X_imp, 'sigma_diff':diff}


    def _em_step(self, Z, r_lower, r_upper, max_workers=1, num_ord_updates=1):
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
        n,p = Z.shape
        assert n>0, 'EM step receives empty input'
        if max_workers ==1:
            args = (Z, r_lower, r_upper, self.sigma, num_ord_updates)
            C, Z_imp, Z = _em_step_body_(args)
            C = C/n
        else:
            if max_workers is None: 
                max_workers = min(32, os.cpu_count()+4)
            divide = n/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [(
                    np.copy(Z[divide[i]:divide[i+1],:]), 
                    r_lower[divide[i]:divide[i+1],:], 
                    r_upper[divide[i]:divide[i+1],:], 
                    self.sigma, num_ord_updates
                    ) for i in range(max_workers)]
            Z_imp = np.empty((n,p))
            C = np.zeros((p,p))
            with ProcessPoolExecutor(max_workers=max_workers) as pool: 
                res = pool.map(_em_step_body_, args)
                for i,(C_divide, Z_imp_divide, Z_divide) in enumerate(res):
                    C += C_divide/n
                    Z_imp[divide[i]:divide[i+1],:] = Z_imp_divide
                    Z[divide[i]:divide[i+1],:] = Z_divide

        sigma = np.cov(Z_imp, rowvar=False) + C 
        sigma = self._project_to_correlation(sigma)
        return sigma, Z_imp, Z

    def _project_to_correlation(self, covariance):
        """
        Projects a covariance to a correlation matrix, normalizing it's diagonal entries. Only checks for diagonal entries to be positive.

        Args:
            covariance (matrix): a covariance matrix

        Returns:
            correlation (matrix): the covariance matrix projected to a correlation matrix
        """
        D = np.diagonal(covariance)
        if any(D==0): 
            raise ZeroDivisionError("unexpected zero covariance for  the latent Z") 
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
        assert all(0<=u_lower[obs_indices]) and all(u_lower[obs_indices] <= u_upper[obs_indices]) and  all(u_upper[obs_indices]<=1)

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

    def back_to_original_order(self):
        #mapping = {}
        num_ord = self.ord_indices.sum()
        seen_cont = seen_ord = 0
        orders = [] 
        for i,index in enumerate(self.cont_indices):
            if index:
                #mapping[seen_cont] = i
                orders.append(seen_cont+num_ord)
                seen_cont += 1
            else:
                orders.append(seen_ord)
                #mapping[num_cont + seen_ord] = i
                seen_ord += 1
        p = len(orders)
        assert len(set(orders))==p and min(orders)==0 and max(orders)==p-1, 'Func back_to_original_order runs into bugs, please report'
        return orders


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




