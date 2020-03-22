'''
A toy program demonstrating application of Metropolis algorithm to sample from a multimodal gaussian mixture.
A background link is here: 
https://stats.stackexchange.com/questions/340010/convergence-issue-in-simple-1d-metropolis-algorithm
'''
import numpy as np
from scipy.stats import norm, rv_histogram
import matplotlib.pyplot as pl


def dist_func(x, mu1, sig1, mu2, sig2):
    v1 = norm.pdf(x, loc=mu1, scale=sig1)
    v2 = norm.pdf(x, loc=mu2, scale=sig2)
    return 0.5 * (v1 + v2)


def compute_analytical(mu1, sig1, mu2, sig2):
    x = np.linspace(-2, 8, 1000)
    y = dist_func(x, mu1, sig1, mu2, sig2)
    return x, y
    

def compute_mcmc(mu1, sig1, mu2, sig2):
    args = mu1, sig1, mu2, sig2    
    num_samples = 30000
    low, high = -10, 10
    # Draw samples
    samples_ = np.full(num_samples, -10.0)
    samples_[0] = 0.0
    sample_idx = 1
    while sample_idx < num_samples:
        u = samples_[sample_idx - 1]
        v = np.random.uniform(low, high)                
        pval = min(1.0, dist_func(v, *args) / dist_func(u, *args))
        if np.random.random() < pval:
            samples_[sample_idx] = v                                                
        else:
            samples_[sample_idx] = u                                                
        sample_idx += 1                        

    return samples_


if __name__ == '__main__':
    mu1, sig1 = 1.0, 1.0
    mu2, sig2 = 4.0, 0.25

    samples = compute_mcmc(mu1, sig1, mu2, sig2)
    x_theory, y_theory = compute_analytical(mu1, sig1, mu2, sig2)    

    # import ipdb; ipdb.set_trace()
    fig, axes = pl.subplots(1, 1)    
    axes.hist(samples, bins=40, histtype='stepfilled', normed=True, color='orange', alpha=0.5)
    axes.plot(x_theory, y_theory, 'b', lw=1)    
    pl.show()
