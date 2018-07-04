import numpy as np
import pandas as pd
import random
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet
import fastcluster as fc
from scipy.spatial.distance import squareform
from abc import ABC, abstractmethod

# Abstract Class - Do not instantiate this class
# Returns all the matrices as a DataFrame
class SingleNmfClass(ABC):
    def __init__(self, x: pd.DataFrame, k: int):
        """Initialize the class and assign vales to class variables

        Keyword Arguments:
            x {dataframe} -- input matrix on which we have to do NMF
            k {int} -- rank for factorization
        """
        if str(type(list(x.values())[0])) == "<class 'pandas.core.frame.DataFrame'>":
            self.x = x.values
        else:
            raise ValueError("Invalid DataType")

        self.k = k

        self.error = float('inf')

        self.eps = np.finfo(list(self.x.values())[0].dtype).eps

    def initialize_variables(self):
        """Initializes consensus matrices. It is run before the iterations of the various trials""" 
        self.consensus_matrix_w = np.zeros((self.x.shape[0], self.x.shape[0]))
        self.consensus_matrix_h = np.zeros((self.x.shape[1], self.x.shape[1]))

    def run(self, trials, iterations, verbose=0):
        """Wrapper function that runs across the different trials
        
        Keyword Arguments:
            verbose {bool} -- for more verbosity (default: {0})
        """
        self.initialize_variables()
        for i in range(0, trials):
            self.initialize_wh()
            self.wrapper_update(iterations, verbose)
            self.calc_connnectivity_matrix(consensus_matrix_w)
            self.calc_connnectivity_matrix(consensus_matrix_h.T)

            if verbose == 1:
                print("\tTrial: %i completed with Error: %f " % (i, self.error))

    def calc_error(self):
        """Calculate error which is the difference between input matrix and product of NMF products"""
        self.error = np.mean(np.abs(self.x - np.dot(self.w, self.h)))
    
    @abstractmethod
    def update_weights(self):
        raise NotImplementedError("Must override update_weights")
    
    @abstractmethod
    def initialize_wh(self):
        raise NotImplementedError("Must override initialize_wh")
