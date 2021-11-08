import numpy as np 
from scipy.stats import random_correlation, norm, expon
import pkg_resources
import pandas as pd

def _cont_to_ord(x, k, rng, by = 'dist', qmin=0.05, qmax=0.95):
    """
    convert entries of x to an ordinal with k levels (0, 1, ..., k-1) using thresholds selected by one choice of the following:
    by = 'dist': select evenly spaced thresholds
    by = 'quantile': select random observations between .05 quantile and .95 quantile as thresholds
    """
    std_dev = np.std(x)
    if by == 'dist':
        # length of k-1 
        # the head and the tail are removed because no value is smaller than the head and 
        # no value is larger than the tail
        cutoffs = np.linspace(np.min(x), np.max(x), k+1)[1:-1]
    elif by == 'quantile':
        # samping cutoffs from 5% quantile to 95% quantile 
        select = (x>np.quantile(x, qmin)) & (x<np.quantile(x, qmax))
        if sum(select)<k:
            raise ValueError(f'Cannot ordinalize the variable because there are fewer than {k} observations from .05 percentile to .95 percentile')
        cutoffs = rng.choice(x[select], k-1, replace = False)
    else:
        raise ValueError('Unsupported cutoff_by option')
    ords = np.digitize(x, np.sort(cutoffs))
    return ords


def cont_to_ord(x, k, by = 'dist', seed=1, max_try=20, qmin=0.05, qmax=0.95, random_generator=None):
    """
    convert entries of x to an ordinal with k levels using thresholds selected by one choice of the following:
    by = 'dist': select evenly spaced thresholds
    by = 'quantile': select random observations between .05 quantile and .95 quantile as thresholds
    by = 'sampling': select random samples from standard normal distribution as thresholds
    """
    if random_generator is None:
        rng = np.random.default_rng(seed)
    else:
        rng = random_generator
    success = False
    for _ in range(max_try):
        result = _cont_to_ord(x,k,rng,by=by,qmin=qmin,qmax=qmax)
        if len(np.unique(result))==k:
            success = True
            break
    if not success:
        print("Failure: the ordinalized variable always has fewer than {k} levels in {max_try} attempts")
        raise 
    return result

def generate_sigma(seed=1, p=15, random_generator=None):
    '''
    Generate a random correlation matrix 
    '''
    if random_generator is None:
        rng = np.random.default_rng(seed)
    else:
        rng = random_generator
    W = rng.normal(size=(p, p))
    covariance = np.matmul(W, W.T)
    D = np.diagonal(covariance)
    D_neg_half = np.diag(1.0/np.sqrt(D))
    return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

def generate_LRGC(var_types, rank, sigma, n=500, ord_num=5, cont_transform=lambda x:x, seed=1, ordinalize_by='dist', random_generator=None):
    cont_index = var_types['cont']
    ord_index = var_types['ord']
    bin_index = var_types['bin']
    all_index = cont_index + ord_index + bin_index
    p = len(all_index)

    if random_generator is None:
        rng = np.random.default_rng(seed)
    else:
        rng = random_generator
    W = rng.normal(size=(p,rank))
    # TODO: check everything of this form with APPLY
    for i in range(W.shape[0]):
        W[i,:] = W[i,:]/np.sqrt(np.sum(np.square(W[i,:]))) * np.sqrt(1 - sigma)
    Z = np.dot(rng.normal(size=(n,rank)), W.T) + rng.normal(size=(n,p), scale=np.sqrt(sigma))
    X_true = Z
    if len(cont_index)>0:
        X_true[:,cont_index] = cont_transform(X_true[:,cont_index])
    for ind in bin_index:
        X_true[:,ind] = cont_to_ord(Z[:,ind], k=2, by=ordinalize_by, random_generator=rng)
    for ind in ord_index:
         X_true[:,ind] = cont_to_ord(Z[:,ind], k=ord_num, by=ordinalize_by, random_generator=rng)
    return X_true, W


def generate_mixed_from_gc(sigma=None, n=2000, seed=1, var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))}, cutoff_by='quantile', random_generator=None):
    '''
    sigma: either a single correlation matrix or a list of correlation matrices
    '''
    if random_generator is None:
        rng = np.random.default_rng(seed)
    else:
        rng = random_generator

    if sigma is None:
        sigma = generate_sigma(p = sum([len(x) for x in var_types.values()]), random_generator=rng)
    if not isinstance(sigma, list):
        sigma = [sigma]
    cont_index = var_types['cont']
    ord_index = var_types['ord']
    bin_index = var_types['bin']
    all_index = cont_index + ord_index + bin_index
    p = len(all_index)
    if min(all_index)!=0 or max(all_index) != (p-1) or len(set(all_index)) != p:
        raise ValueError('Inconcistent specification of variable types indexing')
    if sigma[0].shape[1] != p :
        raise ValueError('Inconcistent dimension between variable lengths and copula correlation')
    X = np.concatenate([rng.multivariate_normal(np.zeros(p), s, size=n) for s in sigma], axis=0)
    # marginal transformation
    X[:,cont_index] = expon.ppf(norm.cdf(X[:,cont_index]), scale = 3)
    for ind in ord_index:
        X[:,ind] = cont_to_ord(X[:,ind], k=5, by=cutoff_by, random_generator=rng)
    for ind in bin_index:
        X[:,ind] = cont_to_ord(X[:,ind], k=2, by=cutoff_by, random_generator=rng)
    return X

def load_GSS(cols=None, to_array = False):
    '''
    Return a subset of General social survey (GSS) datasets selected in year 2014, consisting of 18 variables and 2538 subjects.
    '''
    stream = pkg_resources.resource_stream(__name__, 'data/GSS_2014_18var.csv')
    data = pd.read_csv(stream, index_col=0)
    _cols = data.columns.tolist()
    if cols is not None:  
        try:
            data = data[cols]
        except KeyError:
            print(f'{cols} must be a subset of {_cols}')
            raise
    return np.array(data) if to_array else data

def load_movielens1m_top150():
    stream = pkg_resources.resource_stream(__name__, 'data/movielens1m_top150movie.csv')
    data = pd.read_csv(stream, index_col=0).to_numpy()
    return data


def load_FRED():
    stream = pkg_resources.resource_stream(__name__, 'data/FRED_selected.csv')
    data = pd.read_csv(stream, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data
