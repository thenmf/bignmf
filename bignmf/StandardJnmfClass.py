from lib.JointNmfClass import JointNmfClass
import numpy as np


class StandardNmfClass(JointNmfClass):
    def __init__(self, x: dict, k: int, niter: int, super_niter: int, thresh: float):
        super(StandardNmfClass, self).__init__(x, k, niter, super_niter, thresh)

    def initialize_wh(self):
        number_of_samples = list(self.x.values())[0].shape[0]
        self.w = np.random.rand(number_of_samples, self.k)
        
        self.h = {}
        for key in self.x:
            self.h[key] = np.random.rand(self.k, self.x[key].shape[1])

    def update_weights(self):
        w = self.w
        numerator = np.zeros(w.shape)
        denominator = np.zeros((w.shape[1], w.shape[1]))

        for key, value in self.x.items():
            numerator = numerator + np.dot(self.x[key], self.h[key].T)
            denominator = denominator + np.dot(self.h[key], self.h[key].T)
            self.h[key] = self.h[key] * np.dot(w.T, self.x[key]) / np.dot(np.dot(w.T, w), self.h[key])

        self.w = self.w * numerator / np.dot(w, denominator)
        self.calc_error()
