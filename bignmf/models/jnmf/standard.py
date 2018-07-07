from bignmf.models.jnmf.jnmf_base import JnmfBase
import numpy as np

class StandardJnmf(JnmfBase):
	"""Standard Joint NMF algorithm Class.

	The Joint NMF algorithm first jointly factorizes the multiple input matrices into a common submatrix, ``W`` and several submatrices,
	``Hi`` based on the number of input datasets. For more detailed information on the algorithm, here's the reference to the original paper.
	
		Hong-Qiang Wang et al. jNMFMA: a joint non-negative matrix factorization meta-analysis of transcriptomics data.
		Bioinformatics(2015);31(4):572-580
	
	"""

	def __init__(self, x: dict, k: int):
		"""Initializes the class with Standard NMF algorithm parameters
		
		Args:
			x (dict): Input matrices on which we have to do NMF. Dictionary containing the input matrices as DataFrames. 
					  The common dimension between the matrices should be the row.
			k (int): Rank for factorization
		"""
		super().__init__(x, k)

	def initialize_wh(self):
		"""Initializes the model variables for the Standard JNMF algorithm
			
			Model Variables:
				W: Common submatrix 
				H: Dictionary of submatrices for each of the individual datasets with the same keys as the input dictionary 
		"""
		number_of_samples = list(self.x.values())[0].shape[0]
		self.w = np.random.rand(number_of_samples, self.k)
		self.eps = np.finfo(self.w.dtype).eps
		
		self.h = {}
		for key in self.x:
			self.h[key] = np.random.rand(self.k, self.x[key].shape[1])

	def update_weights(self):
		"""Updates the model variables so that they converge towards :mat:`W.H = X`"""
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
