from models.jnmf.JnmfClass import JointNmfClass
import numpy as np


class StandardNmfClass(JointNmfClass):
    def __init__(self, x: dict, k: int):
        """Initializes the class with Integrative NMF algorithm parameters
        
        Arguments:
            x {dict} -- [Dictionary of the data]
            k {int} -- [Rank]
        """
        super().__init__(x, k)

    def initialize_wh(self):
        number_of_samples = list(self.x.values())[0].shape[0]
        self.w = np.random.rand(number_of_samples, self.k)
        self.eps = np.finfo(self.w.dtype).eps
        
        self.h = {}
        for key in self.x:
            self.h[key] = np.random.rand(self.k, self.x[key].shape[1])

    def update_weights(self):
        w = self.w
        numerator = np.zeros(w.shape)
        denominator = np.zeros(w.shape)

        for key in self.x:
            numerator = numerator + np.dot(self.x[key], self.h[key].T)
            denominator = denominator + np.dot(np.dot(self.w, self.h[key]), self.h[key].T)

        self.w = self.w * numerator / denominator
        #self.w = self.w * numerator / np.dot(w, denominator)

        for key in self.x:
            self.h[key] = np.multiply(self.h[key], np.divide(np.dot(self.w.T, self.x[key]), np.dot(self.w.T, np.dot(self.w, self.h[key]) + self.eps)))
            #self.h[key] = self.h[key] * np.dot(w.T, self.x[key]) / np.dot(np.dot(w.T, w), self.h[key])
        self.calc_error()
