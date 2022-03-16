import numpy as np


class Perceptron(object):
    def __init__(self, lr=0.01, n_iter=10):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for _ in range(self.n_iter):
            for xi, yi in zip(X, y):
                delta = self.lr * (yi - self.predict(xi))
                self.weights += delta * xi
                self.bias += delta

    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0.0, 1, -1)  # 没有使用激活函数
