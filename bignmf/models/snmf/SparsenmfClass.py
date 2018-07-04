from models.snmf.SnmfClass import SingleNmfClass
import numpy as np
import pandas as pd
from sklearn import preprocessing

class SparseNmfClass(SingleNmfClass):
    def __init__(self, x: dict, k: int, spars: int, beta: int):
        super().__init__(x, k)
        self.sparsity = spars
        self.beta = beta

    def initialize_wh(self):
        self.w = np.random.rand(self.x.shape[0], self.k)
        self.eps = np.finfo(self.w.dtype).eps
        self.h = np.random.rand(self.k, self.x.shape[1])
        self.w = preprocessing.normalize(self.w)
        self.a = self.w.dot(self.h)
        self.one = pd.DataFrame(np.ones((self.x .shape[0], 1)))

    def update_weights(self):
        self.h = np.multiply(self.h, np.divide(np.dot(preprocessing.normalize(self.w).T, np.multiply(self.x, np.power(self.a, self.beta - 2))), np.dot(self.w.T, np.power(self.a, self.beta - 1)) + self.sparsity))
        self.a = self.w.dot(self.h)
        self.w = np.multiply(self.w, np.divide(np.dot(np.multiply(np.power(self.a, self.beta - 2), self.x), self.h.T) + np.multiply(preprocessing.normalize(self.w), np.dot(np.dot(self.one, self.one.T), np.multiply(preprocessing.normalize(self.w), np.dot(np.power(self.a, self.beta - 1), self.h.T)))), np.dot(np.power(self.a, self.beta - 1), self.h.T) + np.multiply(preprocessing.normalize(self.w), np.dot(np.dot(self.one, self.one.T), np.multiply(preprocessing.normalize(self.w), np.dot(np.multiply(np.power(self.a, self.beta - 2), self.x), self.h.T))))))
        self.w = preprocessing.normalize(self.w)
        self.a = self.w.dot(self.h)
        self.calc_error()
