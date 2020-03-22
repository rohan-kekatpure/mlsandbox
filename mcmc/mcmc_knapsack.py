'''
Solution of the knapsack problem using Metropolis algorithm
'''

import numpy as np
from copy import deepcopy

def main():
    values = np.array([10, 9, 20, 5])
    weights = np.array([1, 2, 3, 4])
    w_max = 5
    beta = 1.0
    
    assert values.shape == weights.shape
    N = values.shape[0]
    x = np.zeros(N)    
    niter = 100000

    for _ in range(niter):
        idx = np.random.randint(0, N)
        y = x.copy()
        y[idx] = 1 - x[idx]

        w_tot = np.dot(weights, y)
        if w_tot > w_max:
            continue

        val_diff = np.dot(values, y - x)        
        a = min(1, np.exp(beta * val_diff))
        if np.random.random() < a:
            x = y

    print 'x = {0}, value = {1}, weight = {2}'.format(x, np.dot(values, x), np.dot(weights, x))

if __name__ == '__main__':
    main()