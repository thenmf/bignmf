from bignmf.models.snmf.SnmfClass import SingleNmfClass
import numpy as np

class StandardNmfClass(SingleNmfClass):
	def __init__(self, x: dict, k: int):
		"""Initialize the class and assign vales to class variables.

		Args:
			x (dict): input matrix on which we have to do NMF
			k (int): rank for factorization
		"""
		super().__init__(x, k)

	def initialize_wh(self):
		"""Initializes the variables that will be required for Standard NMF.
		"""
		self.w = np.random.rand(self.x.shape[0], self.k)
		self.eps = np.finfo(self.w.dtype).eps
		self.h = np.random.rand(self.k, self.x.shape[1])

	def update_weights(self):
		"""Updates W and H so that they converge in such a way that W.H = X"""
		self.w = np.multiply(self.w, np.divide(np.dot(self.x, self.h.T), np.dot(np.dot(self.w, self.h), self.h.T)))
		self.h = np.multiply(self.h, np.divide(np.dot(self.w.T, self.x), np.dot(self.w.T, np.dot(self.w, self.h) + self.eps)))
		self.calc_error()
