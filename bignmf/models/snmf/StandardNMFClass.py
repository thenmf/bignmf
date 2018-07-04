from models.jnmf.JointNmfClass import JointNmfClass
import numpy as np


class StandardNmfClass(JointNmfClass):
    def __init__(self, x: dict, k: int, niter: int, super_niter: int, thresh: float):
        super(StandardNmfClass, self).__init__(x, k, niter, super_niter, thresh)

    def initialize_wh(self):
        number_of_samples = list(self.x.values())[0].shape[0]
        self.w = np.random.rand(number_of_samples, self.k)
        self.eps = np.finfo(self.W.values.dtype).eps
        
        self.h = np.random.rand(self.k, self.x.shape[1])

    def update_weights(self):
        w = self.w
        numerator = np.zeros(w.shape)
        denominator = np.zeros((w.shape[1], w.shape[1]))

        numerator = np.dot(self.x, self.h.T)
        denominator = np.dot(np.dot(self.w, self.h), self.h.T)

        self.w = self.w * numerator / denominator
        #self.w = self.w * numerator / np.dot(w, denominator)

        self.h = np.multiply(self.h, np.divide(np.dot(self.w.T, self.x), np.dot(self.w.T, np.dot(self.w, self.h) + self.eps)))
        #self.h[key] = self.h[key] * np.dot(w.T, self.x[key]) / np.dot(np.dot(w.T, w), self.h[key])
        self.calc_error()
