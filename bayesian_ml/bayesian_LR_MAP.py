import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as pl


def trace():
    import ipdb; ipdb.set_trace()


def neg_loglik(params, X, Y):
    N = X.shape[0]
    b0, b1, sig = params
    err = Y - np.dot(X, [b0, b1])
    L = np.log(sig) + np.sum(err * err) / (2.0 * sig * sig * N)    
    return L

def map_estimate():
    beta = np.array([1.0, 1.5])
    x = np.linspace(0, 2, 1000)
    ones = np.ones(x.shape)
    X = np.vstack((ones, x)).T    
    noise = np.random.normal(loc=0, scale=0.5, size=x.shape)
    y = np.dot(X, beta) + noise

    params = minimize(neg_loglik, [0.0, 0.0, 0.1], args=(X, y))
    print params.x
    
if __name__ == '__main__':
    map_estimate()