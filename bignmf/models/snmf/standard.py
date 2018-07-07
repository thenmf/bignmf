from bignmf.models.snmf.snmf_base import SnmfBase
import numpy as np
import pandas as pd 

class StandardNmf(SnmfBase):
	'''Standard Single NMF algorithm class

	The algorithm facotrizes the input matrix into two subamtrices :math:`W` and :math:`H`.
	For more detailed information on the algorithm, here's the reference to the original paper.
	
		Lee et al. Learning the parts of objects by non-negative matrix factorization.
		Nature(1999);401:788–791

	'''
	def __init__(self, x: pd.DataFrame, k: int):
		"""Initialize the class and assign vales to class variables.

		Args:
			x (pd.DataFrame): Input matrix
			k (int): Rank for factorization
		"""
		super().__init__(x, k)

	def initialize_wh(self):
		"""Initializes the model variables.
			
		Model Variables:
			W: One submatrix
			H: The other submatrix
		"""
		self.w = np.random.rand(self.x.shape[0], self.k)
		self.eps = np.finfo(self.w.dtype).eps
		self.h = np.random.rand(self.k, self.x.shape[1])

	def update_weights(self):
		"""Updates the model variables so that they converge towards :math:`W.H = X`"""
		self.w = np.multiply(self.w, np.divide(np.dot(self.x, self.h.T), np.dot(np.dot(self.w, self.h), self.h.T)))
		self.h = np.multiply(self.h, np.divide(np.dot(self.w.T, self.x), np.dot(self.w.T, np.dot(self.w, self.h) + self.eps)))
		self.calc_error()
