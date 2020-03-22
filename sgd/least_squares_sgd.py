"""
Least squares model via SGD
"""

import numpy as np
import matplotlib.pylab as pl
from sklearn.linear_model import LinearRegression, SGDRegressor

def ls_sgd(x, y, shuffle=True, epochs=100, lrate=0.01, lrate_decay=1.00):
    # Parameter estimation by SGD
    b0, b1 = 0.5, 0.5    
    eps = 1.0
    b0_evolution = []
    b1_evolution = []    
    data = np.vstack((x, y)).T
    for _ in range(epochs):
        if shuffle:
            np.random.shuffle(data)
        for xi, yi in data:
            gi = 2 * (b0 + b1 * xi - yi)
            b0 -= lrate * gi
            b1 -= lrate * xi * gi
            b0_evolution.append(b0)
            b1_evolution.append(b1)
        lrate *= lrate_decay

    return b0, b1, b0_evolution, b1_evolution


def ls_sklearn(x, y):
    # Parameter estimation by sklearn linear model
    lr = LinearRegression(fit_intercept=True)
    lr.fit(x.reshape((N, 1)), y)
    beta_0_sk = lr.intercept_
    beta_1_sk = lr.coef_[0]
    return beta_0_sk, beta_1_sk

def ls_sklearn_sgd(x, y):
    # Parameter estimation by sklearn SGD
    sgd = SGDRegressor(fit_intercept=True)
    sgd.fit(x.reshape((N, 1)), y)
    beta_0_sk = sgd.intercept_
    beta_1_sk = sgd.coef_[0]
    return beta_0_sk, beta_1_sk

if __name__ == '__main__':
    N = 50
    x = np.linspace(0, 5, N)
    beta_0, beta_1 = 1.0, 2.0
    y = beta_0 + beta_1 * x + np.random.normal(0, 1.0, (N,))

    b0, b1 = ls_sklearn(x, y)
    print "Coefficient estimates [sklearn]: beta0 = %0.3f, beta_1 = %0.3f" % (b0, b1)

    b0, b1 = ls_sklearn_sgd(x, y)
    print "Coefficient estimates [sklearn SGD]: beta0 = %0.3f, beta_1 = %0.3f" % (b0, b1)    

    b0, b1, b0_evol, b1_evol = ls_sgd(x, y, shuffle=True, epochs=50)
    pl.plot(b1_evol, 'm-')
    print "Coefficient estimates [sgd with shuffle, mean]: beta0 = %0.3f, beta_1 = %0.3f" % (np.mean(b0_evol), np.mean(b1_evol))

    b0, b1, b0_evol, b1_evol = ls_sgd(x, y, shuffle=False, epochs=50)
    pl.plot(b1_evol, 'b-')
    print "Coefficient estimates [sgd no shuffle, mean]: beta0 = %0.3f, beta_1 = %0.3f" % (np.mean(b0_evol), np.mean(b1_evol))

    # pl.plot(b0_evolution, 'k-')
    # pl.plot(b1_evol, 'm-')
    pl.show()