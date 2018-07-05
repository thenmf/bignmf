import pandas as pd
import glob
import os

class Datasets():
	"""Class to load and read datasets.
	"""

	@classmethod
	def list_all(cls):
		"""Prints out all the datasets present as CSV in the datasets folder.
		"""
		os.chdir("./datasets")
		l = glob.glob("*.csv")
		li=[x.split('.')[0] for x in l]
		for file in li:
			print(file)
		os.chdir("..")

	@classmethod
	def read(cls, data):
		"""This method reads the dataset specified.

    	Args:
        	data (str): Specifies the path of data to be read.
    	"""
		X = pd.read_csv(r'%s.csv' % (data), index_col=0, header=0, na_values='NaN')
		X = X.fillna(0)
		return X
