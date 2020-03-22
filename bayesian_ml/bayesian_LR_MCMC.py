import numpy as np
from scipy.stats import norm, halfnorm
import matplotlib.pyplot as pl
from copy import deepcopy
from tqdm import tqdm

def trace():
    import ipdb; ipdb.set_trace()


def loglik(state, X, Y):
    M = X.shape[0]
    coefs, sig = state[:-1], state[-1]
    err = Y - np.dot(X, coefs)   
    term1 = -M * np.log(sig)
    term2 = -np.sum(err * err) / (2.0 * sig * sig)    
    return term1 + term2    


def proposalfunc(state):
    alpha, beta1, beta2, sigma = state
    alpha_p, beta1_p, beta2_p = np.random.normal(loc=[alpha, beta1, beta2], scale=0.3)
    sigma_p = -1.0
    while sigma_p < 0:
        sigma_p = norm(sigma, 1.0).rvs()    
    return [alpha_p, beta1_p, beta2_p, sigma_p]    


def logprior(state, distributions):
    return sum(np.log(D.pdf(x)) for x, D in zip(state, distributions))


def generate_data(size):
    alpha, beta1, beta2, sig = 1.0, 1.0, 2.5, 1.0    
    X1 = np.linspace(0, 1.0, size)
    X2 = np.linspace(0, 0.2, size)
    ONES = np.ones(X1.shape)   
    noise = np.random.normal(loc=0, scale=sig, size=size)
    X = np.vstack((ONES, X1, X2)).T
    Y = np.dot(X, [alpha, beta1, beta2]) + noise   
    return X, Y


def plot(chain, skip):
    param_names = ['alpha', 'beta1', 'beta2', 'sigma']
    fig, axes = pl.subplots(4, 2, figsize=(10, 6))
    histargs = dict(normed=True, histtype='step', bins=50, color='r', alpha=0.7)
    for i, pname in zip(range(chain.shape[1]), param_names):
        param = chain[skip:, i]
        axes[i, 0].hist(param, **histargs)        
        axes[i, 1].plot(param, alpha=0.5)        
        axes[i, 0].set_title(pname)
    fig.tight_layout()
    pl.show()


def mcmc_regression():    
    np.random.seed(123) # Uncomment to reproduce the plot exactly
    X, Y = generate_data(100)    
    alpha_dist = beta1_dist = beta2_dist = norm(0, 10)
    sigma_dist = halfnorm(0, 1)
    dists = (alpha_dist, beta1_dist, beta2_dist, sigma_dist)

    nburnin = 50000
    niter = 50000 + nburnin
    ncomps = 4

    chain = np.zeros(shape=(niter, ncomps))
    chain[0, :] = np.abs(np.random.normal(size=ncomps))

    for i in tqdm(range(niter - 1)):                
        v = chain[i]
        log_posterior_old = loglik(v, X, Y) + logprior(v, dists)         
        proposal = proposalfunc(v)
        log_posterior_new = loglik(proposal, X, Y) + logprior(proposal, dists)        
        a = min(0.0, log_posterior_new - log_posterior_old)
        if np.random.random() < np.exp(a):            
            chain[i + 1, :] = proposal
        else:
            chain[i + 1, :] = v

    plot(chain, nburnin)

if __name__ == '__main__':
    mcmc_regression()