from bignmf.models.jnmf.JnmfClass import JointNmfClass
import numpy as np

class StandardNmfClass(JointNmfClass):
	def __init__(self, x: dict, k: int):
		"""Initializes the class with Integrative NMF algorithm parameters
		
		Args:
			x (dict): Input matrices on which we have to do NMF
			k (int): Rank for factorization
		"""
		super().__init__(x, k)

	def initialize_wh(self):
		"""Initializes the variables that will be required for Standard NMF."""
		number_of_samples = list(self.x.values())[0].shape[0]
		self.w = np.random.rand(number_of_samples, self.k)
		self.eps = np.finfo(self.w.dtype).eps
		
		self.h = {}
		for key in self.x:
			self.h[key] = np.random.rand(self.k, self.x[key].shape[1])

	def update_weights(self):
		"""Updates W and H so that they converge in such a way that W.H = X"""
		w = self.w
		numerator = np.zeros(w.shape)
		denominator = np.zeros(w.shape)

		for key in self.x:
			numerator = numerator + np.dot(self.x[key], self.h[key].T)
			denominator = denominator + np.dot(np.dot(self.w, self.h[key]), self.h[key].T)

		self.w = self.w * numerator / denominator

		for key in self.x:
			self.h[key] = np.multiply(self.h[key], np.divide(np.dot(self.w.T, self.x[key]), np.dot(self.w.T, np.dot(self.w, self.h[key]) + self.eps)))
		self.calc_error()
