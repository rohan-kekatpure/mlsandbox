import numpy as np
import matplotlib.pylab as pl
from sklearn.ensemble import GradientBoostingRegressor
import pdb 

if __name__ == "__main__":
    # Generate synthetic non-linear data
    x = np.linspace(0, 10, 100).reshape((-1, 1))
    y = 2 * x * np.exp(-x * 0.5) + 0.2 * np.random.random(x.shape)

    # Gradient boosting step    
    gbr = GradientBoostingRegressor(loss="ls", 
                                    learning_rate=0.2, 
                                    n_estimators=100,
                                    max_depth=2)
    gbr.fit(x, y)
    yPred = gbr.predict(x)
    pl.plot(x, y, "o", color="gray")
    pl.plot(x, yPred, "-")
    pl.show()
