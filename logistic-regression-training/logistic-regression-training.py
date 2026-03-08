import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0

    for i in range(steps):
        linear_model = np.dot(X,w) + b
        p =_sigmoid(linear_model)

        gradientW = np.dot(X.T, (p-y))/n_samples
        gradientB = np.mean(p-y)

        w = w -lr*gradientW
        b = b -lr*gradientB

    return (w,b)