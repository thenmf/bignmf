import numpy as np
import pandas as pd
import random
from abc import ABC, abstractmethod
from bignmf.models.nmf import Nmf

# Abstract Class - Do not instantiate this class
# Returns all the matrices as a DataFrame
class JointNmfClass(Nmf):
	def __init__(self, x: dict, k: int):
		"""Initialize the class and assign vales to class variables and all joint nmf classes inherit from this.

		Args:
			x (dict): input matrices on which we have to do NMF
			k (int): rank for factorization
		"""
		super().__init__(k)
		if str(type(list(x.values())[0])) == "<class 'pandas.core.frame.DataFrame'>":
			self.column_index={}
			self.x={}
			self.row_index=list(list(x.values())[0].index)
			for key in x:
				self.column_index[key] = list(x[key].columns)
				self.x[key] = x[key].values
				if all(self.row_index != x[key].index):
					raise ValueError("Row indices are not uniform")
		else:
			raise ValueError("Invalid DataType")
		self.error = float('inf')
		self.eps = np.finfo(list(self.x.values())[0].dtype).eps

	def initialize_variables(self):
		"""Initializes all the variables except the w and h. It is run before the iterations of the various trials.""" 
		number_of_samples = list(self.x.values())[0].shape[0]
		self.consensus_matrix_w = np.zeros((number_of_samples, number_of_samples))
		self.consensus_matrix_h = {}
		for key in self.x:
			number_of_features = self.x[key].shape[1]
			self.consensus_matrix_h[key] = np.zeros((number_of_features, number_of_features))
	
	def run(self, trials, iterations, verbose=0):
		"""Runs the NMF algorithm for the specified iterations over the specified trials
		
		Args:
			trials (int}: Number of different trials.
			iterations (int}: Number of iterations.
			verbose (bool): To increase verbosity. Defaults to 0.
		"""
		self.initialize_variables()
		for i in range(0, trials):
			self.initialize_wh()
			self.wrapper_update(iterations, verbose if i==0 else 0) 
			self.consensus_matrix_w += self.connectivity_matrix(self.w, axis=1)
			for key in self.h:
				self.consensus_matrix_h[key] += self.connectivity_matrix(self.h[key], axis=0)
			if verbose == 1:
				print("\tSuper iteration: %i completed with Error: %f " % (i, self.error))
		# Normalization
		self.consensus_matrix_w = self.reorder_consensus_matrix(self.consensus_matrix_w / trials)
		for key in self.h:
			self.consensus_matrix_h[key] /= trials
		# Converting values to DataFrames
		class_list = ["class-%i" % a for a in list(range(self.k))]
		self.w = pd.DataFrame(self.w, index=self.row_index, columns=class_list)
		self.h = {k: pd.DataFrame(self.h[k], index=class_list, columns=self.column_index[k]) for k in self.h}

	def calc_cophenetic_correlation(self):    
		"""Calculated the cophentic correlation co-efficients and stores it in the instance
		"""
		self.cophenetic_correlation_w = self.cophenetic_correlation(self.consensus_matrix_w)
		self.cophenetic_correlation_h = {}
		for key, cmh in self.consensus_matrix_h.items():
			self.cophenetic_correlation_h[key] = self.cophenetic_correlation(cmh)

	def cluster_data(self):
		"""Clusters the output matrices, W and the other H matrices
		"""
		self.w_cluster = self.cluster_matrix(self.w, 1)
		self.h_cluster = {}
		for key, val in self.h.items():
			self.h_cluster[key] = self.cluster_matrix(val,0)

	def calc_consensus_matrices(self):
		"""Makes the final consensus matrix by reordering the consensus matrix
		"""
		self.consensus_matrix_w = self.reorder_consensus_matrix(self.consensus_matrix_w)
		for key, cmh in self.consensus_matrix_h.items():
			self.consensus_matrix_h[key] = self.reorder_consensus_matrix(cmh)

	# TODO - Add docstring for the formulae
	def calc_error(self):
		"""Calculates the euclidean distance error with the following formulae.
		"""
		self.error = 0
		for key in self.x:
			self.error += np.mean(np.abs(self.x[key] - np.dot(self.w, self.h[key])))
	
	@abstractmethod
	def update_weights(self):
		raise NotImplementedError("Must override update_weights")
	
	@abstractmethod
	def initialize_wh(self):
		raise NotImplementedError("Must override initialize_wh")
