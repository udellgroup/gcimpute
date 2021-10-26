import numpy as np
from scipy.stats import norm, truncnorm

def _em_step_body_(args):
    """
    Does a step of the EM algorithm, needed to dereference args to support parallelism
    """
    return _em_step_body(*args)

def _em_step_body(Z, r_lower, r_upper, sigma, num_ord_updates):
    """
    Iterate the rows over provided matrix 
    """
    n, p = Z.shape
    Z_imp = np.copy(Z)
    C = np.zeros((p,p))
    C_ord = np.zeros((n,p))
    trunc_warn = False
    loglik = 0
    for i in range(n):
        c, z_imp, z, c_ordinal, _loglik, warn = _em_step_body_row(Z[i,:], r_lower[i,:], r_upper[i,:], sigma, num_ord_updates)
        Z_imp[i,:] = z_imp
        Z[i,:] = z
        C_ord[i,:] = c_ordinal
        C += c
        loglik += _loglik
        trunc_warn = trunc_warn or warn
    # TO DO: no need to return Z, just edit it during the process
    if trunc_warn:
        print('Bad truncated normal stats appear, suggesting the existence of outliers. We skipped the outliers now. More stable version to come...')
    return C, Z_imp, Z, C_ord, loglik


def _em_step_body_row(Z_row, r_lower_row, r_upper_row, sigma, num_ord_updates):
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

    Returns:
        C (matrix): results in the updated covariance when added to the empircal covariance
        Z_imp_row (array): Z_row with latent ordinals updated and missing entries imputed 
        Z_row (array): input Z_row with latent ordinals updated
    """
    missing_indices = np.isnan(Z_row)
    obs_indices = ~missing_indices

    # obtain correlation sub-matrices
    # obtain submatrices by indexing a "cartesian-product" of index arrays
    sigma_obs_obs = sigma[np.ix_(obs_indices,obs_indices)]
    sigma_obs_missing = sigma[np.ix_(obs_indices, missing_indices)]
    sigma_missing_missing = sigma[np.ix_(missing_indices, missing_indices)]

    if any(missing_indices):
        tot_matrix = np.concatenate((np.identity(len(sigma_obs_obs)), sigma_obs_missing), axis=1)
        intermed_matrix = np.linalg.solve(sigma_obs_obs, tot_matrix)
        sigma_obs_obs_inv = intermed_matrix[:, :len(sigma_obs_obs)]
        J_obs_missing = intermed_matrix[:, len(sigma_obs_obs):]
    else:
        sigma_obs_obs_inv = np.linalg.solve(sigma_obs_obs, np.identity(len(sigma_obs_obs)))


    # OBSERVED ORDINAL ELEMENTS
    # when there is an observed ordinal to be imputed and another observed dimension, impute this ordinal
    # The update here only requires the observed variables, but needs to use the relative location of 
    # ordinal variables in all observed variables. 
    # TO DO: the updating order of ordinal variables may make a difference in the algorithm statbility
    # initialize vector of variances for observed ordinal dimensions
    p, num_ord = Z_row.shape[0], r_upper_row.shape[0]
    # var_ordinal stores the conditional variance of each variable given all other observation,
    # it has 0 at observed continuous and missing locations.
    var_ordinal = np.zeros(p)  
    ord_obs_indices = (np.arange(p) < num_ord) & obs_indices
    truncnorm_warn = False
    if sum(obs_indices) >=2 and any(ord_obs_indices):
        for update_iter in range(num_ord_updates):
            # used to efficiently compute conditional mean
            sigma_obs_obs_inv_Zobs_row = np.dot(sigma_obs_obs_inv, Z_row[obs_indices])
            j_in_obs = 0

            # TO DO: accelerate is possible. Replace the for-loop with vector/matrix computation.
            # Essentially, replace the Gauss-Seidel style update with a Jacobi style update for the nonlinear system
            for j in range(num_ord):
                # j is the location in the p-dim coordinate
                # j_in_obs is the location of j in the obs-dim coordiate
                # TODO simplify the iteration command into a single for-loop
                if obs_indices[j]:
                    new_var_ij = (1.0/sigma_obs_obs_inv[j_in_obs, j_in_obs].item())
                    new_std_ij = np.sqrt(new_var_ij)
                    new_mean_ij = Z_row[j] - new_var_ij*sigma_obs_obs_inv_Zobs_row[j_in_obs]
                    a_ij, b_ij = (r_lower_row[j] - new_mean_ij) / new_std_ij, (r_upper_row[j] - new_mean_ij) / new_std_ij
                    try:
                        _mean, _var = truncnorm.stats(a=a_ij,b=b_ij,loc=new_mean_ij,scale=new_std_ij,moments='mv')
                        if np.isfinite(_var):
                            var_ordinal[j] = _var
                        if np.isfinite(_mean):
                            Z_row[j] = _mean
                    except RuntimeWarning:
                        #print(f'Bad truncated normal stats: lower {r_lower_row[j]}, upper {r_upper_row[j]}, a {a_ij}, b {b_ij}, mean {new_mean_ij}, std {new_std_ij}')
                        truncnorm_warn = True
                    # update the relative location after we see an ordinal observed variable
                    j_in_obs += 1

    # initialize C 
    C = np.diag(var_ordinal)

    # MISSING ELEMENTS
    Z_obs = Z_row[obs_indices]
    Z_imp_row = np.copy(Z_row)
    if any(missing_indices):
        # impute missing entries
        Z_imp_row[missing_indices] = np.matmul(J_obs_missing.T,Z_obs) 
        # expected covariance in the missing dimensions
        C[np.ix_(missing_indices, missing_indices)] += sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing)
        # expected covariance brought due to the oridnal entries
        if np.sum(var_ordinal) > 0: 
            ord_in_obs = np.arange(sum(ord_obs_indices))
            cov_missing_obs_ord = J_obs_missing[ord_in_obs].T * var_ordinal[ord_obs_indices]
            C[np.ix_(missing_indices, ord_obs_indices)] += cov_missing_obs_ord
            C[np.ix_(ord_obs_indices, missing_indices)] += cov_missing_obs_ord.T
            C[np.ix_(missing_indices, missing_indices)] += np.matmul(cov_missing_obs_ord, J_obs_missing[ord_in_obs])

    # log-likelihood at observed locations
    negloglik = np.linalg.slogdet(sigma_obs_obs)[1] + np.inner(np.dot(sigma_obs_obs_inv, Z_obs), Z_obs)
    loglik = -negloglik/2.0 
    return C, Z_imp_row, Z_row, var_ordinal, loglik, truncnorm_warn


def _LRGC_em_row_step_body_(args):
    """
    Does a step of the EM algorithm, needed to dereference args to support parallelism
    """
    return _LRGC_em_row_step_body(*args)

def _LRGC_em_row_step_body(Z, r_lower, r_upper, U, d, sigma, num_ord_updates):
    """
    Iterate the rows over provided matrix 
    """
    n, p = Z.shape
    rank = U.shape[1]

    A = np.zeros((n, rank, rank))
    SS = np.copy(A)
    S = np.zeros((n,rank))
    C_ord = np.zeros((n,p))
    trunc_warn = False
    loglik = 0
    s = 0

    for i in range(n):
        zi, Ai, si, ssi, c_ordinal, _loglik, _s, warn = _LRGC_em_row_step_body_row(Z[i,:], r_lower[i,:], r_upper[i,:], U, d, sigma, num_ord_updates)
        Z[i] = zi
        A[i] = Ai
        S[i] = si
        SS[i] = ssi
        C_ord[i] = c_ordinal
        loglik += _loglik
        s += _s
        trunc_warn = trunc_warn or warn
    if trunc_warn:
        print('Bad truncated normal stats appear, suggesting the existence of outliers. We skipped the outliers now. More stable version to come...')
    return Z, A, S, SS, C_ord, loglik, s


def _LRGC_em_row_step_body_row(Z_row, r_lower_row, r_upper_row, U, d, sigma, num_ord_updates=1):
    rank = U.shape[1]
    missing_indices = np.isnan(Z_row)
    obs_indices = ~missing_indices
    zi_obs = Z_row[obs_indices]
    Ui_obs = U[obs_indices,:]
    UU_obs = np.dot(Ui_obs.T, Ui_obs) 

    # used in both ordinal and factor block
    res = np.linalg.solve(UU_obs + sigma * np.diag(1.0/np.square(d)), np.concatenate((np.identity(rank), Ui_obs.T),axis=1))
    Ai = res[:,:rank]
    AU = res[:,rank:]

    p, num_ord = Z_row.shape[0], r_upper_row.shape[0]
    var_ordinal = np.zeros(p)  
    ord_obs_indices = (np.arange(p) < num_ord) & obs_indices
    truncnorm_warn = False
    if sum(obs_indices) >=2 and any(ord_obs_indices):
        for _ in range(num_ord_updates):
            mu = (zi_obs - np.dot(Ui_obs, np.dot(AU, zi_obs)))/sigma
            j_in_obs = 0

            for j in range(num_ord):
                if obs_indices[j]:
                    new_var_ij  = sigma/(1 - np.dot(U[j,:].T, np.dot(Ai, U[j,:])))
                    new_std_ij = np.sqrt(new_var_ij)
                    new_mean_ij = Z_row[j] - new_var_ij*mu[j_in_obs]
                    a_ij, b_ij = (r_lower_row[j] - new_mean_ij) / new_std_ij, (r_upper_row[j] - new_mean_ij) / new_std_ij
                    try:
                        _mean, _var = truncnorm.stats(a=a_ij,b=b_ij,loc=new_mean_ij,scale=new_std_ij,moments='mv')
                        if np.isfinite(_var):
                            var_ordinal[j] = _var
                        if np.isfinite(_mean):
                            Z_row[j] = _mean
                    except RuntimeWarning:
                        #print(f'Bad truncated normal stats: lower {r_lower_row[j]}, upper {r_upper_row[j]}, a {a_ij}, b {b_ij}, mean {new_mean_ij}, std {new_std_ij}')
                        truncnorm_warn = True
                    # update the relative location after we see an ordinal observed variable
                    j_in_obs += 1

    si = np.dot(AU, zi_obs)
    ssi = np.dot(AU * var_ordinal[obs_indices], AU.T) + np.outer(si, si.T)

    negloglik = np.log(sigma) * p + np.linalg.slogdet(np.identity(rank) + np.outer(d/sigma, d) * UU_obs)[1]
    negloglik += np.sum(zi_obs**2) - np.dot(zi_obs.T, np.dot(Ui_obs, si))
    loglik = -negloglik/2.0

    zi_obs_norm = np.power(zi_obs, 2).sum()
    return Z_row, Ai, si, ssi, var_ordinal, loglik, zi_obs_norm, truncnorm_warn

def _LRGC_em_col_step_body_(args):
    """
    Does a step of the EM algorithm, needed to dereference args to support parallelism
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
        _w, _s = _LRGC_em_col_step_body_col(Z[:,j], C_ord[:,j], U[j], sigma, A, S, SS)
        W[j] = _w
        s += _s
    return W, s

def _LRGC_em_col_step_body_col(Z_col, C_ord_col, U_col, sigma, A, S, SS):
    index_j = ~np.isnan(Z_col)
    # numerator
    rj = _sum_2d_scale(M=S, c=Z_col, index=index_j) + np.dot(_sum_3d_scale(A, c=C_ord_col, index=index_j), U_col)
    # denominator
    Fj = _sum_3d_scale(SS+sigma*A, c=np.ones(A.shape[0]), index = index_j) 
    w_new = np.linalg.solve(Fj,rj) 
    s = np.dot(rj, w_new)
    return w_new, s


def _sum_3d_scale(M, c, index):
    res = np.empty((M.shape[1], M.shape[2]))
    for j in range(M.shape[1]):
        for k in range(M.shape[2]):
            res[j,k] = np.sum(M[index,j,k] * c[index])
    return res

def _sum_2d_scale(M, c, index):
    res = np.empty(M.shape[1])
    for j in range(M.shape[1]):
        res[j] = np.sum(M[index,j] * c[index])
    return res
