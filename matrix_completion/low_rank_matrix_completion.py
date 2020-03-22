from scipy.linalg import diagsvd
import numpy as np
from numpy.linalg import svd, norm
from itertools import product
import matplotlib.pylab as pl


def get_error(X, Y):
    norm_XmY = norm(np.nan_to_num(X - Y))
    norm_Y = norm(np.nan_to_num(Y))
    return norm_XmY / max(1.0, norm_Y)


def compare_plot(I1, I2):
    fig, axes = pl.subplots(1, 2)
    axes[0].imshow(I1, cmap='gray')
    axes[1].imshow(I2, cmap='gray')
    pl.savefig('img/reconstructed.png')
    pl.show()


def fillmat(M):
    m, n = M.shape
    X = np.zeros(shape=(m, n))    
    tau = 1.0    
    mu_min = 1.0e-8
    eta_mu = 0.25
    mu = eta_mu * norm(np.nan_to_num(M)) 

    niter = 0
    max_iter = 10000
    xtol = 1.0e-3

    while (mu > mu_min) and (niter < max_iter):
        delta = 1.0
        while delta > xtol:         
            X_prev = X
            Y = X - tau * np.nan_to_num(X - M)
            U, S, V = svd(Y, full_matrices=False)
            S1 = np.maximum(S - tau * mu, 0)
            S1 = diagsvd(S1, n, n)
            X = np.dot(U, np.dot(S1, V))
            delta = get_error(X, X_prev)            

        mu = max(mu * eta_mu, mu_min)
        niter += 1       
        print 'mu = {:0.4e}'.format(mu)

    return X


if __name__ == '__main__':
    # m, n, r = 32, 32, 5
    # P = np.random.normal(size=(m, r))
    # Q = np.random.normal(size=(r, n))
    # M0 = np.dot(P, Q)
    # # M0 = np.array(np.random.uniform(size=(m, n)))
    # M1 = np.copy(M0)

    from PIL import Image
    M0 = np.array(Image.open('img/face.jpg').convert('L'), dtype=float)
    M1 = np.copy(M0)
    m, n = M0.shape
    print M0.shape

    # Change random elements of M to nan
    null_frac = 0.8
    p = int(np.round(null_frac * m * n))

    rc_pairs = list(product(range(m), range(n)))    
    rnd_idx = np.array(np.random.permutation(rc_pairs)).T
    take_idx = rnd_idx[:, :p].tolist()
    M1[take_idx] = np.nan

    X = fillmat(M1)
    print get_error(X, M0)

    compare_plot(M1, X)
    

