import numpy as np
import pandas as pd
import random
from abc import ABC, abstractmethod
from bignmf.models.nmf import Nmf

# Abstract Class - Do not instantiate this class
# Returns all the matrices as a DataFrame
class SingleNmfClass(Nmf):
	def __init__(self, x: pd.DataFrame, k: int):
		"""Initialize the class and assign vales to class variables and all single nmf classes inherit from this.

		Args:
			x (dataframe): input matrix on which we have to do NMF
			k (int): rank for factorization
		"""
		super().__init__(k)
		self.row_index = list(x.index)
		self.column_index = list(x)
		self.x = x.values
		self.error = float('inf')

	def initialize_variables(self):
		"""Initializes consensus matrices. It is run before the iterations of the various trials""" 
		self.consensus_matrix_w = np.zeros((self.x.shape[0], self.x.shape[0]))
		self.consensus_matrix_h = np.zeros((self.x.shape[1], self.x.shape[1]))

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
			self.consensus_matrix_h += self.connectivity_matrix(self.h, axis=0)
			if verbose == 1:
				print("\tTrial: %i completed with Error: %f " % (i, self.error))
		# Normalization
		self.consensus_matrix_w = self.reorder_consensus_matrix(self.consensus_matrix_w / trials)
		self.consensus_matrix_h = self.reorder_consensus_matrix(self.consensus_matrix_h / trials)
		# Converting values to DataFrames
		class_list = ["class-%i" % a for a in list(range(self.k))]
		self.w = pd.DataFrame(self.w, index=self.row_index, columns=class_list)
		self.h = pd.DataFrame(self.h, index=class_list, columns=self.column_index)

	def calc_error(self):
		"""Calculates the euclidean distance error with the following formulae.
		"""
		self.error = 0
		self.error = np.mean(np.abs(self.x - np.dot(self.w, self.h)))

	def calc_cophenetic_correlation(self):    
		"""Calculated the cophentic correlation co-efficients and stores it in the instance.
		"""
		self.cophenetic_correlation_w = self.cophenetic_correlation(self.consensus_matrix_w)
		self.cophenetic_correlation_h = self.cophenetic_correlation(self.consensus_matrix_h)

	def cluster_data(self):
		"""Clusters the output matrices, W and the other H matrices.
		"""
		self.w_cluster = self.cluster_matrix(self.w, 1)
		self.h_cluster = self.cluster_matrix(self.h, 0)

	def calc_consensus_matrices(self):
		self.consensus_matrix_w = self.reorder_consensus_matrix(self.consensus_matrix_w)
		self.consensus_matrix_h = self.reorder_consensus_matrix(self.consensus_matrix_h)
	
	@abstractmethod
	def update_weights(self):
		raise NotImplementedError("Must override update_weights")
	
	@abstractmethod
	def initialize_wh(self):
		raise NotImplementedError("Must override initialize_wh")
