from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet
import fastcluster as fc
from scipy.spatial.distance import squareform

class NmfBase(ABC):
	"""Base abstract NMF class.
	All Joint NMF and Single NMF classes inherit from this class.   
	"""
	def __init__(self, k):
		"""Initalisation for the Base NMF class
		
		Args:
			k (int): rank for the NMF classification
		"""
		self.k = k
	
	def wrapper_update(self, iter, verbose=0):
		"""Updates the weights for the number of iterations specified.
		
		Args:
			iter (int): Number of iterations through which the model should update.
			verbose (int, optional): Defaults to 0. To increase verbsosity.
		"""
		for i in range(1, iter):
			self.update_weights()
			if verbose == 1 and i % 1 == 0:
				print("\t\titer: %i | error: %f" % (i, self.error))

   # TODO - Fix the hack 
	@classmethod
	def connectivity_matrix(cls, x:np.array, axis):
		"""Calculates the connecitivity matrix along the given axis.

		The function first classifies the input matrix along the specified axis and then matrix multiplies it to itself
		keeping the rank-side common.
		
		Args:
			x (np.array): the input matrix.
			axis (int): axis along which the connectivity matrix should be classified. For W matrix, it's 1 
			and for the H matrix it is 0.

		Returns:
			np.array: the output matrix.
		"""
		
		if axis == 0:
			a = cls.classify_by_max(x, 0)
			return np.dot(a.T, a)
		elif axis == 1:
			a = cls.classify_by_max(x.T, 0).T
			return np.dot(a, a.T)

	@staticmethod    
	def reorder_consensus_matrix(M: np.array):
		"""Reoders the consensus matrix.
		
		Args:
			M (np.array): Input matrix
		
		Returns:
			np.array: Reordered output matrix
		"""

		M = pd.DataFrame(M)
		Y = 1 - M
		Z = linkage(squareform(Y), method='average')
		ivl = leaves_list(Z)
		ivl = ivl[::-1]
		reorderM = pd.DataFrame(M.values[:, ivl][ivl, :], index=M.columns[ivl], columns=M.columns[ivl])
		return reorderM.values

	@staticmethod
	def cophenetic_correlation(consensus_matrix):
		"""Calculates the cophentic correlation co-efficient from a consensus matrix.
		
		Arguments:
			consensus_matrix (np.array): the unordered consensus matrix
		
		Returns:
			int: the cophenetic correlation co-efficient
		"""
		ori_dists = fc.pdist(consensus_matrix)
		Z = fc.linkage(ori_dists, method='average')
		[coph_corr, temporary] = cophenet(Z, ori_dists)
		return coph_corr

	@staticmethod    
	def classify_by_max(x: np.array, axis):
		"""Classifies the input array based on max filter along the specified axis
		
		Args:
			x (np.array): Input array
			axis (int): Axis along which to perform the classification
		
		Returns:
			np.array: Classified array in one-hot format
		"""
		return (x == np.amax(x, axis=axis)).astype(float)

	@classmethod
	def cluster_matrix(cls, x: np.array, axis):
		"""Clusters the provided matrix along the given axis
		
		Args:
			x (np.array): Input matrix
			axis (int): Axis along which to perform the clustering
		
		Returns:
			np.array: Output matrix
		"""
		a = cls.classify_by_max(x, axis)
		return a.T.sort_values(by=list(a.index)).T
