import numpy as np
from scipy.optimize import root
from scipy.stats import norm
from .helper_core import *

def get_cat_index_freq(X_cat):
    p_cat = X_cat.shape[1]
    freq = {j:[] for j in range(p_cat)}
    nlevel = np.zeros(p_cat)
    for j in range(p_cat):
        x = observed_part(X_cat[:,j])
        _, counts = np.unique(x, return_counts=True)
        freq[j] = counts/sum(counts)
        nlevel[j] = len(counts)
    return freq, nlevel


def init_mu(prob):
    k = len(prob)
    _prob = prob[0] / (np.array(prob[1:]) + prob[0])
    return -np.sqrt(2) * norm.ppf(_prob)

def softmax_colwise(X):
    '''
    Compute the softmax of X in each row over columns
    '''
    X = (X.T - X.max(axis=1)).T
    m = np.exp(X)
    rowsum = m.sum(axis=1)
    try:
        r = (m.T / rowsum).T
    except RuntimeWarning:
        print(rowsum.min(), rowsum.max())
        raise
    return r

def E_softmax_MC_with_jacobian(Mu, beta, nMC = 5000, seed=11):
    Mu = np.array(Mu)
    k = len(Mu)
    # fix samples at each call
    if seed is not None:
        np.random.seed(seed)
    z= np.random.standard_normal(size = (nMC,k)) + Mu
    #
    pi_x = softmax_colwise(z*beta)
    val = pi_x.mean(axis=0).flatten()
    # ([n,k,k] - [n,1,k]) * [n,k,1] = [n,k,k] * [n,k,1]
    jac = (np.tile(np.eye(k), (nMC,1,1)) - pi_x[:, None, :]) * pi_x[...,np.newaxis]
    jac = jac.mean(axis=0) * beta
    return val, jac


def get_solve_mu(prob, beta, nMC = 5000, seed=1):
    '''
    return the error function of mu, parameterized by prob, rho and beta
    '''
    try:
        assert np.isclose(np.array(prob).sum(), 1)
    except Exception:
        print(prob)
        raise
        
    k = len(prob)
    
    def solve_mu_MC(mu):
        mu = np.array([0] + list(mu))
        val, jacobian = E_softmax_MC_with_jacobian(mu, beta, nMC=nMC, seed=seed)
        error = val-prob
        return error[1:], jacobian[1:,1:]
    
    return solve_mu_MC

def solve_nominal_mu(prob, beta = 1000, nMC = 5000, seed = 1, eps = 1e-4):
    best_precis = 100
    best_r = None
    
    mu0 = init_mu(prob)
    inits = np.zeros((4, len(mu0)))
    for i,e in enumerate([0, 1, 3]):
        inits[i,:] = mu0/(2**e)
    l = inits.shape[0]
    
    for muinit in inits:
        fsolve = get_solve_mu(prob, beta, nMC, seed=seed)
        r = root(fsolve, muinit, method = 'hybr', jac = True)
        precis = np.abs(r.fun).max()
        if precis < best_precis:
            best_precis = precis
            best_r = r
        if best_precis < eps:
            break
    
    return best_r

def get_cat_mu(freqs, beta = 1000, nMC = 5000, seed = 1, eps = 1e-4, verbose=False, prob_est=False, **kwargs):
    d = dict_values_len(freqs)
    lf = len(freqs)
    mu = np.zeros(d)
    fs = np.zeros(d)
    probs = {i:[] for i in freqs} 
    K_cats = {i:[] for i in freqs} 
    est_precis = np.zeros(lf)
    start = 0
    for i, (idx, freq) in enumerate(freqs.items()):
        out = solve_nominal_mu(freq, beta = beta, nMC = nMC, seed = seed, eps = eps)
        mui = out.x
        dmui = len(mui)
        index = start + 1 +  np.arange(dmui)
        mu[index] = mui
        fs[index] = out.fun
        est_precis[i] = np.abs(out.fun).max()
        if prob_est:
        	probs[idx] = E_argmax_MC(Mu = np.append([0], mui), **kwargs)
        	K_cats[idx] = (dmui+1) * np.ones(dmui+1)
        start = start + 1 + len(mui)
    return {'mu':mu, 'f_root':fs, 'est_precis':est_precis, 'probs':probs, 'K_cats':K_cats}

def E_argmax_MC(Mu, Sigma=None, n=10000, seed=111):
    '''
    MC estimation of Prob(z_k=max_i z_i) for z from N(Mu, Sigma)
    '''
    if Sigma is None:
    	Sigma = np.identity(len(Mu))
    Mu = np.array(Mu)
    Sigma = np.array(Sigma)
    k = len(Mu)
    rng = np.random.default_rng(seed)
    z=rng.multivariate_normal(mean=Mu, cov=Sigma, size=n)
    prob = np.bincount(z.argmax(axis=1), minlength=len(Mu))/n
    return prob  