from lib.functions import reorderConsensusMatrix, calc_cophenetic_correlation
import numpy as np
import pandas as pd
import random

def classify_by_max(x: np.array):
    return (x == np.amax(x, axis=0)).astype(float)


def classify_by_z(x: np.array, thresh):
    a = (x - np.mean(x, axis=1).reshape((-1, 1))) / (np.std(x, axis=1)).reshape((-1, 1))
    classification = np.zeros(a.shape)
    classification[a > thresh] = 1
    return classification


# Abstract Class - Do not instantiate this class
# Returns all the matrices as a DataFrame
class JointNmfClass:
    def __init__(self, x: dict, k: int, niter: int, super_niter: int, thresh: float):
        if str(type(list(x.values())[0])) == "<class 'pandas.core.frame.DataFrame'>":
            self.column_index={}
            self.x={}
            self.row_index=list(random.choice(list(x.values())).index)
            for key in x:
                self.column_index[key] = list(x[key].columns)
                self.x[key] = x[key].values
                if all(self.row_index != x[key].index):
                    raise ValueError("Row indices are not uniform")
        else:
            raise ValueError("Invalid DataType")

        self.k = k
        self.niter = niter
        self.super_niter = super_niter
        self.cmw = None
        self.w = None
        self.h = None
        self.thresh = thresh
        self.error = float('inf')
        self.eps = np.finfo(list(self.x.values())[0].dtype).eps

    def initialize_variables(self):
        """Initializes all the variables except the w and h. It is run before the iterations of the various trials""" 
        number_of_samples = list(self.x.values())[0].shape[0]

        self.cmw = np.zeros((number_of_samples, number_of_samples))
        self.max_class = {}
        self.max_class_cm = {}
        self.z_score = {}
        self.coph_corr_w = None
        self.coph_corr_h = {}
        for key in self.x:
            number_of_features = self.x[key].shape[1]
            self.max_class[key] = np.zeros((self.k, number_of_features))
            self.max_class_cm[key] = np.zeros((number_of_features, number_of_features))
            self.coph_corr_h[key] = None

    def wrapper_update(self, verbose=0):
        for i in range(1, self.niter):
            self.update_weights()
            if verbose == 1 and i % 1 == 0:
                print("\t\titer: %i | error: %f" % (i, self.error))
    
    def super_wrapper(self, verbose=0):
        """Wrapper function that runs across the different trials
        
        Keyword Arguments:
            verbose {bool} -- for more verbosity (default: {0})
        """
        self.initialize_variables()
        for i in range(0, self.super_niter):
            self.initialize_wh()
            self.wrapper_update(verbose if i==0 else 0) 
            self.cmw += self.connectivity_matrix_w()
            for key in self.h:
                connectivity_matrix = lambda a: np.dot(a.T, a)
                self.max_class_cm[key] += connectivity_matrix(classify_by_max(self.h[key]))

            if verbose == 1:
                print("\tSuper iteration: %i completed with Error: %f " % (i, self.error))



        # Normalization
        self.cmw = reorderConsensusMatrix(self.cmw / self.super_niter)
        
        #Cophenetic Correlation
        self.coph_corr_w = calc_cophenetic_correlation(self.cmw)
        for key, cmh in self.max_class_cm.items():
            self.coph_corr_h[key] = calc_cophenetic_correlation(cmh)

        #Reordering Consensus Matrix
        for key in self.h:
            self.max_class_cm[key] /= self.super_niter
            self.max_class_cm[key] = reorderConsensusMatrix(self.max_class_cm[key])

        # Classification
        for key, val in self.h.items():
            self.max_class[key] = classify_by_max(val)

        # Converting values to DataFrames
        class_list = ["class-%i" % a for a in list(range(self.k))]
        self.w = pd.DataFrame(self.w, index=self.row_index, columns=class_list)

        self.h = self.conv_dict_np_to_df(self.h)
        self.max_class = self.conv_dict_np_to_df(self.max_class)

    def conv_dict_np_to_df(self, a: dict):
        """[Converts the passed 'h' like attribute from numpy to dataframe ]
        'h' like attributes are dictionaries with the keys being the names of the different passed datasets

        Arguments:
            a {dict} -- [the 'h' like variable to be converted]
        
        Returns:
            [dict] -- [the converted variable]
        """ 
        class_list = ["class-%i" % a for a in list(range(self.k))]
        return {k: pd.DataFrame(a[k], index=class_list, columns=self.column_index[k]) for k in a}

    # TODO - invalid value ocurred
    def calc_z_score(self):
        for key in self.h:
            self.z_score[key] = (self.h[key] - np.mean(self.h[key], axis=1).reshape((-1, 1))) / (
                    self.eps + np.std(self.h[key], axis=1).reshape((-1, 1)))

    def connectivity_matrix_w(self):
        max_tiled = np.tile(self.w.max(1).reshape((-1, 1)), (1, self.w.shape[1]))
        max_index = np.zeros(self.w.shape)
        max_index[self.w == max_tiled] = 1
        return np.dot(max_index, max_index.T)

    def calc_error(self):
        self.error = 0
        for key in self.x:
            self.error += np.mean(np.abs(self.x[key] - np.dot(self.w, self.h[key])))

    def update_weights(self):
        raise NotImplementedError("Must override update_weights")

    def initialize_wh(self):
        raise NotImplementedError("Must override initialize_wh")
