import numpy as np
from .truncnorm_comp import *
new_ord_update = True

def _latent_operation_body_(args):
    """
    Dereference args to support parallelism
    """
    # if the last element in args is a dict, then it is the dict with additional keyword arguments
    if isinstance(args[-1], dict):
        return  _latent_operation_body(*args[:-1], **args[-1])
    else:
        return _latent_operation_body(*args)

def _latent_operation_body(task, Z, r_lower, r_upper, sigma, num_ord_updates, ord_indices, has_truncation, **kwargs):
    '''
    Args:
        task: str in ['em', 'fillup', 'sample']
    Returns:
        out_dict:
            If task == 'em', 
                out_dict has keys ['Z', 'var_ordinal', 'Z_imp', 'loglik', 'C'].
            If task == 'fillup',
                out_dict has keys ['var_ordinal', 'Z_imp'].
            If task == 'sample',
                out_dict has keys ['Z_imp_sample']
    '''
    n, p = Z.shape
    out_dict = {}
    trunc_warn = 0

    if task == 'em':
        out_dict['var_ordinal'] = np.zeros((n,p))
        out_dict['Z_imp'] = Z.copy()
        out_dict['Z'] = Z
        out_dict['loglik'] = 0
        out_dict['C'] = np.zeros((p,p))
    elif task == 'fillup':
        out_dict['Z_imp'] = Z.copy()
        out_dict['var_ordinal'] = np.zeros((n,p))
    elif task == 'sample':
        try:
            num = kwargs['num']
            seed = kwargs['seed']
        except KeyError:
            print('Additional arguments of num and seed need to be provided to sample latent row')
            raise 
        out_dict['Z_imp_sample'] = np.empty((n,p,num))
    else:
        print(f'invalid task type: {task}')
        raise 

    for i, Z_row in enumerate(Z):
        if has_truncation:
            false_ord = np.isclose(r_lower[i], r_upper[i]) 
            true_ord = ~false_ord
            # adjust r_lower and r_upper
            r_lower_row, r_upper_row = r_lower[i][true_ord], r_upper[i][true_ord]
            # adjust ord_indices
            ord_indices_input = ord_indices.copy()
            int_ord_indices = np.flatnonzero(ord_indices)
            ord_indices_input[int_ord_indices[false_ord]] = False
        else:
            r_lower_row, r_upper_row = r_lower[i], r_upper[i]
            ord_indices_input = ord_indices
        row_out_dict = _latent_operation_row(task, Z_row, r_lower_row, r_upper_row, sigma, num_ord_updates, ord_indices_input, **kwargs)
        trunc_warn += row_out_dict['truncnorm_warn']
        if task == 'em':
            for key in ['Z_imp', 'Z', 'var_ordinal']:
                out_dict[key][i] = row_out_dict[key]
            for key in ['loglik', 'C']:
                out_dict[key] += row_out_dict[key]
        elif task == 'fillup':
            for key in ['Z_imp', 'var_ordinal']:
                out_dict[key][i] = row_out_dict[key]
        elif task == 'sample':
            missing_indices = np.isnan(Z_row)
            if any(missing_indices):
                out_dict['Z_imp_sample'][i, missing_indices, :] = row_out_dict['Z_imp_sample']
        else:
            pass

    if trunc_warn>0:
        print(f'Bad truncated normal stats appear {trunc_warn} times, suggesting the existence of outliers. We skipped the outliers now. More stable version to come...')

    return out_dict

def _latent_operation_row(task, Z_row, r_lower_row, r_upper_row, sigma, num_ord_updates, ord_indices, **kwargs):
    '''
    Arguments
    ---------
        task: str in ['em', 'fillup', 'sample']

    Returns
    -------
        out_dict:
            If task == 'em', 
                out_dict has keys ['truncnorm_warn', 'Z', 'var_ordinal', 'Z_imp', 'loglik', 'C'].
            If task == 'fillup',
                out_dict has keys ['truncnorm_warn', 'var_ordinal', 'Z_imp'].
            If task == 'sample',
                out_dict has keys ['truncnorm_warn', 'Z_imp_sample']

            var_ordinal: array of shape (n_features,)
                The conditional variance due to truncation, i.e. E(z|a < z < b).
                Zero at continuous entries, nonzero at ordinal entries

            C: array of shape (n_features, n_features)
                The conditional co-variance due to missingness, i.e. E(z_i z_j) where both of z_i, z_j may be observed or missing 
    '''
    if task == 'sample':
        try:
            num = kwargs['num']
            seed = kwargs['seed']
        except KeyError:
            print('Additional arguments of num and seed need to be provided to sample latent row')
            raise 
    out_dict = {'truncnorm_warn':False}

    missing_indices = np.isnan(Z_row)
    obs_indices = ~missing_indices
    # special cases
    if task == 'sample' and all(obs_indices):
        out_dict['Z_imp_sample'] = None
        return out_dict

    # prepare indices and quantities
    ord_obs_indices = ord_indices & obs_indices
    ord_in_obs = ord_obs_indices[obs_indices]
    obs_in_ord = ord_obs_indices[ord_indices]

    out = _prepare_quantity_GC(missing_indices, sigma)
    sigma_obs_obs_inv = out['sigma_obs_obs_inv']
    sigma_obs_obs = out['sigma_obs_obs']
    sigma_missing_missing = out['sigma_missing_missing']
    J_obs_missing = out['J_obs_missing']
    sigma_obs_missing = out['sigma_obs_missing']

    # conduct ordinal truncated mean & var approximation
    sigma_obs_obs_inv_Zobs_row_func = lambda z_row_obs, sigma_obs_obs_inv=sigma_obs_obs_inv: np.dot(sigma_obs_obs_inv, z_row_obs)
    Z_row, var_ordinal, truncnorm_warn = _update_z_row_ord(z_row=Z_row, 
                                                           r_lower_row=r_lower_row, 
                                                           r_upper_row=r_upper_row, 
                                                           num_ord_updates=num_ord_updates,
                                                           sigma_obs_obs_inv_Zobs_row_func = sigma_obs_obs_inv_Zobs_row_func,
                                                           sigma_obs_obs_inv_diag=np.diag(sigma_obs_obs_inv),
                                                           obs_indices = obs_indices,
                                                           ord_obs_indices = ord_obs_indices,
                                                           ord_in_obs = ord_in_obs,
                                                           obs_in_ord = obs_in_ord 
                                                           )
    out_dict['truncnorm_warn'] = truncnorm_warn

    if task == 'em':
        out_dict['Z'] = Z_row
        out_dict['var_ordinal'] = var_ordinal
    if task == 'fillup':
        out_dict['var_ordinal'] = var_ordinal

    Z_obs = Z_row[obs_indices]
    # fill up the missing entries at Z_row
    if task in ['em', 'fillup']:
        Z_imp_row = Z_row.copy()
        if any(missing_indices):
            Z_imp_row[missing_indices] = np.matmul(J_obs_missing.T,Z_obs) 
        out_dict['Z_imp'] = Z_imp_row

    # sample the missing entries 
    # TODO: also sample the ordinal entries
    if task == 'sample':
        np.random.seed(seed)
        cond_mean = np.matmul(J_obs_missing.T,Z_obs)
        cond_cov = sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing)
        Z_imp_num = np.random.multivariate_normal(mean=cond_mean, cov=cond_cov, size=num)
        # Z_imp_num has shape (n_mis, num)
        Z_imp_num = Z_imp_num.T
        out_dict['Z_imp_sample'] = Z_imp_num

    # compute log-likelihood at observed entries
    if task == 'em':
        p = len(Z_row)
        negloglik = np.linalg.slogdet(sigma_obs_obs)[1] + np.inner(np.dot(sigma_obs_obs_inv, Z_obs), Z_obs) + p*np.log(2*np.pi)
        loglik = -negloglik/2.0 
        out_dict['loglik'] = loglik

    # compute the conditional covariance
    if task == 'em':
        # initialize C 
        C = np.diag(var_ordinal)

        if any(missing_indices):
            # expected covariance in the missing dimensions
            C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing)
            # expected covariance brought due to the oridnal entries
            if var_ordinal.sum() > 0: 
                cov_missing_obs_ord = J_obs_missing[ord_in_obs].T * var_ordinal[ord_obs_indices]
                C[np.ix_(missing_indices, ord_obs_indices)] += cov_missing_obs_ord
                C[np.ix_(ord_obs_indices, missing_indices)] += cov_missing_obs_ord.T
                C[np.ix_(missing_indices, missing_indices)] += np.matmul(cov_missing_obs_ord, J_obs_missing[ord_in_obs])
        out_dict['C'] = C
  
    return out_dict

def _prepare_quantity_GC(missing_indices, sigma):
    obs_indices = ~missing_indices

    # obtain correlation sub-matrices
    # obtain submatrices by indexing a "cartesian-product" of index arrays
    sigma_obs_obs = sigma[np.ix_(obs_indices,obs_indices)]
    sigma_obs_missing = sigma[np.ix_(obs_indices, missing_indices)]
    sigma_missing_missing = sigma[np.ix_(missing_indices, missing_indices)]

    out = {'sigma_missing_missing':sigma_missing_missing, 'sigma_obs_missing':sigma_obs_missing, 'sigma_obs_obs':sigma_obs_obs}

    if any(missing_indices):
        tot_matrix = np.concatenate((np.identity(len(sigma_obs_obs)), sigma_obs_missing), axis=1)
        intermed_matrix = np.linalg.solve(sigma_obs_obs, tot_matrix)
        sigma_obs_obs_inv = intermed_matrix[:, :len(sigma_obs_obs)]
        J_obs_missing = intermed_matrix[:, len(sigma_obs_obs):]
    else:
        sigma_obs_obs_inv = np.linalg.solve(sigma_obs_obs, np.identity(len(sigma_obs_obs)))
        J_obs_missing = None
    out['J_obs_missing'] = J_obs_missing
    out['sigma_obs_obs_inv'] = sigma_obs_obs_inv

    return out

def _LRGC_latent_operation_body_(args):
    """
    Does a step of the EM algorithm, needed to dereference args to support parallelism
    """
    if isinstance(args[-1], dict):
        return  _LRGC_latent_operation_body(*args[:-1], **args[-1])
    else:
        return _LRGC_latent_operation_body(*args)

def _LRGC_latent_operation_body(task, Z, r_lower, r_upper, U, d, sigma, num_ord_updates, ord_indices, has_truncation, **kwargs):
    '''
    Args:
        task: str in ['em', 'fillup', 'sample']
    Returns:
        out_dict:
            If task == 'em', 
                out_dict has keys ['Z', 'var_ordinal', 'loglik', 'A', 's', 'ss', 'zobs_norm'].
            If task == 'fillup',
                out_dict has keys ['var_ordinal', 'Z_imp'].
            If task == 'sample',
                out_dict has keys ['Z_imp_sample']
    '''
    #print(Z.shape)
    n, p = Z.shape
    rank = U.shape[1]

    out_dict = {}
    trunc_warn = 0

    if task == 'em':
        out_dict['Z'] = Z
        out_dict['var_ordinal'] = np.zeros((n,p))
        out_dict['loglik'] = 0
        out_dict['A'] = np.zeros((n, rank, rank))
        out_dict['s'] = np.zeros((n, rank))
        out_dict['ss'] = np.zeros((n, rank, rank))
        out_dict['zobs_norm'] = 0
    elif task == 'fillup':
        out_dict['Z_imp'] = Z.copy()
        out_dict['var_ordinal'] = np.zeros((n,p))
    elif task == 'sample':
        try:
            num = kwargs['num']
            seed = kwargs['seed']
        except KeyError:
            print('Additional arguments of num and seed need to be provided to sample latent row')
            raise 
        out_dict['Z_imp_sample'] = np.empty((n,p,num))
    else:
        print(f'invalid task type: {task}')
        raise 

    for i, Z_row in enumerate(Z):
        if has_truncation:
            false_ord = np.isclose(r_lower[i], r_upper[i]) 
            true_ord = ~false_ord
            # adjust r_lower and r_upper
            r_lower_row, r_upper_row = r_lower[i][true_ord], r_upper[i][true_ord]
            # adjust ord_indices
            ord_indices_input = ord_indices.copy()
            int_ord_indices = np.flatnonzero(ord_indices)
            ord_indices_input[int_ord_indices[false_ord]] = False
        else:
            r_lower_row, r_upper_row = r_lower[i], r_upper[i]
            ord_indices_input = ord_indices

        row_out_dict = _LRGC_latent_operation_row(task, Z_row, r_lower_row, r_upper_row, U, d, sigma, num_ord_updates, ord_indices_input, **kwargs)
        trunc_warn += row_out_dict['truncnorm_warn']
        if task == 'em':
            for key in ['Z', 'var_ordinal', 'A', 's', 'ss']:
                out_dict[key][i] = row_out_dict[key]
            for key in ['loglik', 'zobs_norm']:
                out_dict[key] += row_out_dict[key]
        elif task == 'fillup':
            for key in ['Z_imp', 'var_ordinal']:
                out_dict[key][i] = row_out_dict[key]
        elif task == 'sample':
            missing_indices = np.isnan(Z_row)
            if any(missing_indices):
                out_dict['Z_imp_sample'][i, missing_indices, :] = row_out_dict['Z_imp_sample']
        else:
            pass

    if trunc_warn>0:
        print(f'Bad truncated normal stats appear {trunc_warn} times, suggesting the existence of outliers. We skipped the outliers now. More stable version to come...')

    return out_dict

def _LRGC_latent_operation_row(task, Z_row, r_lower_row, r_upper_row, U, d, sigma, num_ord_updates, ord_indices, **kwargs):
    '''
    Arguments
    ---------
        task: str in ['em', 'fillup', 'sample']

    Returns
    -------
        out_dict:
            If task == 'em', 
                out_dict has keys ['truncnorm_warn', 'Z', 'var_ordinal', 'loglik', 'A', 's', 'ss', 'zobs_norm'].
            If task == 'fillup',
                out_dict has keys ['truncnorm_warn', 'var_ordinal', 'Z_imp'].
            If task == 'sample',
                out_dict has keys ['truncnorm_warn', 'Z_imp_sample']

            var_ordinal: array of shape (n_features,)
                The conditional variance due to truncation, i.e. E(z|a < z < b).
                Zero at continuous entries, nonzero at ordinal entries


    '''
    if task == 'sample':
        try:
            num = kwargs['num']
            seed = kwargs['seed']
        except KeyError:
            print('Additional arguments of num and seed need to be provided to sample latent row')
            raise 
    out_dict = {'truncnorm_warn':False}

    p, rank = U.shape
    missing_indices = np.isnan(Z_row)
    obs_indices = ~missing_indices
    # special cases
    if task == 'sample' and all(obs_indices):
        out_dict['Z_imp_sample'] = None
        return out_dict
    ord_obs_indices = ord_indices & obs_indices
    
    # Ui_obs: |O_i| * k
    Ui_obs = U[obs_indices,:]
    # matrix multiplication, |O_i|k^2 
    # UU_obs: k by k
    UU_obs = np.dot(Ui_obs.T, Ui_obs) 

    # used in both ordinal and factor block
    # linear system, k^3+|O_i|k^2 
    res = np.linalg.solve(UU_obs + sigma * np.diag(1.0/np.square(d)), np.concatenate((np.identity(rank), Ui_obs.T),axis=1))
    # Ai: k by k
    Ai = res[:,:rank]
    # AU: k by |O_i|
    AU = res[:,rank:]

    # Each execution: k|O_i|
    sigma_obs_obs_inv_Zobs_row_func = lambda z_obs, U_obs=Ui_obs, AU=AU, sigma=sigma: (z_obs - np.dot(U_obs, np.dot(AU, z_obs)))/sigma
    # |O_i|*k^2
    sigma_obs_obs_inv_diag = (1 - np.einsum('ij, jk, ki -> i', Ui_obs, Ai, Ui_obs.T))/sigma

    zi_obs = Z_row[obs_indices]
    # p_ord + k|O_i|
    Z_row, var_ordinal, truncnorm_warn = _update_z_row_ord(z_row=Z_row, 
                                                           r_lower_row=r_lower_row, 
                                                           r_upper_row=r_upper_row, 
                                                           num_ord_updates=num_ord_updates,
                                                           sigma_obs_obs_inv_Zobs_row_func = sigma_obs_obs_inv_Zobs_row_func,
                                                           sigma_obs_obs_inv_diag = sigma_obs_obs_inv_diag,
                                                           obs_indices = obs_indices,
                                                           ord_obs_indices = ord_obs_indices,
                                                           ord_in_obs = ord_obs_indices[obs_indices],
                                                           obs_in_ord = ord_obs_indices[ord_indices]
                                                           )
    out_dict['truncnorm_warn'] = truncnorm_warn
    
    zi_obs = Z_row[obs_indices] 
    si = np.dot(AU, zi_obs)

    if task == 'fillup':
        out_dict['var_ordinal'] = var_ordinal
        Z_imp_row = Z_row.copy()
        Z_imp_row[missing_indices] = np.dot(U[missing_indices,:], si)
        out_dict['Z_imp'] = Z_imp_row
    elif task == 'sample':
        np.random.seed(seed)
        Ui_mis = U[missing_indices,:]
        cond_mean = np.dot(Ui_mis, si)
        p_mis = Ui_mis.shape[0]
        cond_cov = sigma * (np.identity(p_mis) + np.matmul(np.matmul(Ui_mis, Ai), Ui_mis.T))
        Z_imp_num = np.random.multivariate_normal(mean=cond_mean, cov=cond_cov, size=num)
        # Z_imp_num has shape (n_mis, num)
        Z_imp_num = Z_imp_num.T
        out_dict['Z_imp_sample'] = Z_imp_num
    else:
        out_dict['var_ordinal'] = var_ordinal
        out_dict['Z'] = Z_row
        out_dict['A'] = Ai
        out_dict['s'] = si
        # k|O_i| + k^2|O_i| + k^2 = k^2|O_i|
        out_dict['ss'] = np.dot(AU * var_ordinal[obs_indices], AU.T) + np.outer(si, si.T)

        num_obs = obs_indices.sum()
        negloglik = p*np.log(2*np.pi)
        negloglik += np.log(sigma) * (num_obs-rank) + np.linalg.slogdet(np.identity(rank) * sigma + np.power(d,2) * UU_obs)[1]
        negloglik += (np.sum(zi_obs**2) - np.dot(zi_obs.T, np.dot(Ui_obs, si)))/sigma
        out_dict['loglik'] = -negloglik/2.0
        out_dict['zobs_norm'] = np.power(zi_obs, 2).sum()
  
    return out_dict


def _LRGC_em_col_step_body_(args):
    """
    Dereference args to support parallelism
    """
    return _LRGC_em_col_step_body(*args)


def _LRGC_em_col_step_body(Z, C_ord, U, sigma, A, S, SS):
    """
    Iterate the rows over provided matrix 
    """
    W = np.zeros_like(U)
    p = Z.shape[1]
    s = 0
    for j in range(p):
        index_j = ~np.isnan(Z[:,j])
        _w, _s = _LRGC_em_col_step_body_col(Z[index_j,j], C_ord[index_j,j], U[j], sigma, A[index_j,:,:], S[index_j], SS[index_j,:,:])
        W[j] = _w
        s += _s
    return W, s

def _LRGC_em_col_step_body_col(Z_col, C_ord_col, U_col, sigma, A, S, SS):
    # numerator
    # rj = _sum_2d_scale(M=S, c=Z_col, index=index_j) + np.dot(_sum_3d_scale(A, c=C_ord_col, index=index_j), U_col)
    rj = np.einsum('ij, i -> j', S, Z_col)
    rj += np.dot(np.einsum('ijk, i -> jk', A, C_ord_col), U_col)
    # denominator
    # Fj = _sum_3d_scale(SS+sigma*A, c=np.ones(A.shape[0]), index = index_j) 
    Fj = np.einsum('ijk -> jk', SS+sigma*A)
    w_new = np.linalg.solve(Fj,rj) 
    s = np.dot(rj, w_new)
    return w_new, s


def _update_z_row_ord(z_row, r_lower_row, r_upper_row, 
                      num_ord_updates, 
                      sigma_obs_obs_inv_Zobs_row_func, 
                      sigma_obs_obs_inv_diag, 
                      obs_indices, ord_obs_indices, ord_in_obs, obs_in_ord):
    '''
    For a row, modify the conditional mean and compute the conditional var at ordinal entries.
    Computation complexity: num_ord_updates * (M1 + p_ord * M2), 
    where M1 denotes the computation of executing sigma_obs_obs_inv_Zobs_row_func once, 
    and M2 denotes the computation of evaluating a truncated normal stats

    Arguments
    ---------
        sigma_obs_obs_inv_Zobs_row_func: A function with (n_obs, ) narray-like input and (n_obs, ) narray-like output 
            Input: z_obs 
            Output: sigma_obs_obs_inv_Zobs_row 
            The matrix-vector product Sigma_{obs, obs}^{-1} * z_{obs}
        sigma_obs_obs_inv_diag: array of shape (nobs, )
            The diagonal of Sigma_{obs, obs}^{-1} 

    Returns
    -------
        var_ordinal: array of shape (n_features,)
            The conditional variance due to truncation, i.e. E(z|a < z < b).
            Zero at continuous entries, nonzero at ordinal entries
    '''
    p, num_ord = z_row.shape[0], r_upper_row.shape[0]
    var_ordinal = np.zeros(p)

    truncnorm_warn = False
    # when there is an observed ordinal to be imputed and another observed dimension, impute this ordinal
    # The update here only requires the observed variables, but needs to use the relative location of 
    # ordinal variables in all observed variables. 
    # TO DO: the updating order of ordinal variables may make a difference in the algorithm statbility
    # initialize vector of variances for observed ordinal dimensions
    if sum(obs_indices) >=2 and any(ord_obs_indices):
        #ord_obs_iter = np.arange(num_ord)[ord_obs_indices[:num_ord]]
        ord_obs_iter = np.flatnonzero(ord_obs_indices)
        ord_in_obs_iter = np.flatnonzero(ord_in_obs)
        obs_in_ord_iter = np.flatnonzero(obs_in_ord)

        #assert len(ord_obs_iter) == len(ord_in_obs_iter) == len(obs_in_ord_iter)
        
        for _ in range(num_ord_updates):
            sigma_obs_obs_inv_Zobs_row = sigma_obs_obs_inv_Zobs_row_func(z_row[obs_indices])
            # Essentially, replace the Gauss-Seidel style update with a Jacobi style update for the nonlinear system
            # ord_obs_iter has sum(ord_obs_indices) entries, ord_in_obs has sum(ord_obs_indices[obs_indices]).
            # Provided the True entries in ord_obs_indices are all in obs_indices,
            # ord_obs_iter and ord_in_obs have the same length
            new_std = np.sqrt(1.0/sigma_obs_obs_inv_diag[ord_in_obs_iter])
            new_mean = z_row[ord_obs_iter] - (new_std**2) * sigma_obs_obs_inv_Zobs_row[ord_in_obs_iter]
            a, b = r_lower_row[obs_in_ord_iter], r_upper_row[obs_in_ord_iter]  
            out =  get_truncnorm_moments_vec(a = a, b= b, mu = new_mean, std = new_std)
            _mean, _std =  out['mean'], out['std']
            old_mean, old_std = z_row[ord_obs_iter], var_ordinal[ord_obs_iter]
            loc  = ~np.isfinite(_mean)
            _mean[loc] = old_mean[loc]
            z_row[ord_obs_iter] = _mean
            loc  = ~np.isfinite(_std)
            _std[loc] = 0
            var_ordinal[ord_obs_iter] = _std**2
            ''' archived old slow implementation
            for j_in_ord, j_in_obs, j in zip(obs_in_ord_iter, ord_in_obs_iter, ord_obs_iter):
                # j is the location in the p-dim coordinate
                # j_in_obs is the location of j in the obs-dim coordinate
                # j_in_ord is the lcoation of j in the ord-dim coordinate
                new_var_ij = 1.0/sigma_obs_obs_inv_diag[j_in_obs]
                new_std_ij = np.sqrt(new_var_ij)
                new_mean_ij = z_row[j] - new_var_ij* sigma_obs_obs_inv_Zobs_row[j_in_obs]
                a_ij, b_ij = r_lower_row[j_in_ord], r_upper_row[j_in_ord]
                #_mean, _var = truncnorm.stats(a=a_ij,b=b_ij,loc=new_mean_ij,scale=new_std_ij,moments='mv')
                _mean, _std = get_truncnorm_moments(a = a_ij, b = b_ij, mu = new_mean_ij, std = new_std_ij)
                if np.isfinite(_std) and _std>0:
                    var_ordinal[j] = _std**2
                if np.isfinite(_mean):
                    z_row[j] = _mean
            '''

    return z_row, var_ordinal, truncnorm_warn
