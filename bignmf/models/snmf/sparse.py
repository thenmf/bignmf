from bignmf.models.snmf.snmf_base import SnmfBase
import numpy as np
import pandas as pd
from sklearn import preprocessing

class SparseNmf(SnmfBase):
	'''
	The update rule used in Sparse NMF has been taken from the paper Roux, J.L. (2015). Sparse NMF – half-baked or well done ?.
	'''
	def __init__(self, x: pd.DataFrame, k: int, spars: int, beta: int):
		"""Initialize the class and assign vales to class variables.

		Args:
			x (pd.DataFrame): input matrix on which we have to do NMF
			k (int): rank for factorization
			spars (int): sparsity factor
			beta (int): exponential factor
		"""
		super().__init__(x, k)
		self.sparsity = spars
		self.beta = beta

	def initialize_wh(self):
		"""Initializes the variables that will be required for Sparse NMF.
		"""
		self.w = np.random.rand(self.x.shape[0], self.k)
		self.eps = np.finfo(self.w.dtype).eps
		self.h = np.random.rand(self.k, self.x.shape[1])
		self.w = preprocessing.normalize(self.w)
		self.a = self.w.dot(self.h)
		self.one = pd.DataFrame(np.ones((self.x .shape[0], 1)))

	def update_weights(self):
		"""Updates W,H and A so that they converge in such a way that W.H = X"""
		self.h = np.multiply(self.h, np.divide(np.dot(preprocessing.normalize(self.w).T, np.multiply(self.x, np.power(self.a, self.beta - 2))), np.dot(self.w.T, np.power(self.a, self.beta - 1)) + self.sparsity))
		self.a = self.w.dot(self.h)
		self.w = np.multiply(self.w, np.divide(np.dot(np.multiply(np.power(self.a, self.beta - 2), self.x), self.h.T) + np.multiply(preprocessing.normalize(self.w), np.dot(np.dot(self.one, self.one.T), np.multiply(preprocessing.normalize(self.w), np.dot(np.power(self.a, self.beta - 1), self.h.T)))), np.dot(np.power(self.a, self.beta - 1), self.h.T) + np.multiply(preprocessing.normalize(self.w), np.dot(np.dot(self.one, self.one.T), np.multiply(preprocessing.normalize(self.w), np.dot(np.multiply(np.power(self.a, self.beta - 2), self.x), self.h.T))))))
		self.w = preprocessing.normalize(self.w)
		self.a = self.w.dot(self.h)
		self.calc_error()
