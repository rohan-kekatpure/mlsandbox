import numpy as np


class Sigmoid:
    @staticmethod
    def f(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def d(x):
        v = Sigmoid.f(x)
        return v * (1. - v)


class Relu:
    EPS = 1.e-12

    @staticmethod
    def f(x):
        return np.maximum(0, x)

    @staticmethod
    def d(x):
        return np.where(x < Relu.EPS, 0, 1)


class Error_MSE:
    @staticmethod
    def f(y1, y2):
        pass

    @staticmethod
    def d(y1, y2):
        return np.mean(y1 - y2, axis=1).reshape(-1, 1)


class Error_CrossEntropy:
    @staticmethod
    def f(y, f):
        pass

    @staticmethod
    def d(y, f):
        pass


class MLP:
    def __init__(self, in_dim, out_dim, hidden_layer_sizes, activation, error, learning_rate):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.error_func = error
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.wmtx = []
        self.bmtx = []

        # Initialize weight matrices and bias vectors
        self._init()

    def _init(self):
        """
        Initialize weight and bias matrices with standard normal random numbers
        """
        hsize = self.hidden_layer_sizes

        # First layer
        self.wmtx.append(np.random.normal(size=(hsize[0], self.in_dim)))
        self.bmtx.append(np.random.normal(size=(hsize[0], 1)))

        # Hidden layers
        for i in range(1, self.n_hidden_layers):
            wl = np.random.normal(size=(hsize[i], hsize[i - 1]))
            bl = np.random.normal(size=(hsize[i], 1))
            self.wmtx.append(wl)
            self.bmtx.append(bl)

        # Output layer
        self.wmtx.append(np.random.normal(size=(self.out_dim, hsize[-1])))
        self.bmtx.append(np.random.normal(size=(self.out_dim, 1)))

    def forward(self, x):
        """
        Forward pass for a single input vector x
        """
        w = self.wmtx
        b = self.bmtx
        sigma = self.activation.f
        activations = [sigma(w[0] @ x + b[0])]
        for i in range(1, self.n_hidden_layers + 1):
            z = w[i] @ activations[i - 1] + b[i]
            activations.append(sigma(z))

        return activations

    def predict(self, x):
        return self.forward(x)[-1]

    def Z(self, w, b, a):
        return w @ a + b

    def _fit_batch(self, xb, yb):
        """
        Backpropagation on single batch
        """
        batch_size = xb.shape[1]
        w = self.wmtx
        b = self.bmtx
        deriv = self.activation.d

        # Compute all activations for the forward pass
        activations = self.forward(xb)

        # Compute error for the output layer
        L = self.n_hidden_layers + 1   # Hidden layers + output layer
        zL = self.Z(w[L - 1], b[L - 1], activations[L - 2])
        deltas = [None] * L
        deltas[L - 1] = self.error_func.d(activations[L - 1], yb) * deriv(zL)

        # Compute deltas for layers 2..L-1
        for l in range(L - 2, 0, -1):
            zl = w[l] @ activations[l - 1] + b[l]
            deltas[l] = (w[l + 1].T @ deltas[l + 1]) * deriv(zl)

        # Compute for the first hidden layer
        z0 = w[0] @ xb + b[0]
        deltas[0] = (w[1].T @ deltas[1]) * deriv(z0)

        # Compute gradient of input-layer weights
        wgrads = [None] * L
        bgrads = [None] * L

        wgrads[0] = np.tensordot(deltas[0], xb, axes=((1, ), (1, ))) / batch_size
        bgrads[0] = deltas[0].mean(axis=1).reshape(-1, 1)

        # Gradient of hidden layer weights
        for l in range(1, L):
            wgrads[l] = np.tensordot(deltas[l], activations[l - 1], axes=((1, ), (1, ))) / batch_size
            bgrads[l] = deltas[l].mean(axis=1).reshape(-1, 1)

        # Gradient descent update
        for l in range(L):
            w[l] -= self.learning_rate * wgrads[l]
            b[l] -= self.learning_rate * bgrads[l]

    def fit(self, x, y):
        pass


def make_mlp():
    in_dim = 5
    out_dim = 5
    hidden_layer_sizes = (100, )
    mlp = MLP(in_dim, out_dim, hidden_layer_sizes, Sigmoid, Error_MSE, 0.001)
    return mlp


def fwd_iter(mlp, x, n_iter):
    for k in range(n_iter):
        print('iteration -> {}, x -> {}'.format(k, x.T))
        x = mlp.forward(x)[-1]


if __name__ == '__main__':
    mlp = make_mlp()
    x = np.random.normal(size=(5, 1))
