import pandas as pd
import glob
import os
from pathlib import Path as pth

class Datasets():
	"""Class to load and read datasets.
	"""

	@classmethod
	def list_all(cls):
		"""Prints out all the datasets present as CSV in the datasets folder.
		"""
		for x in pth(__file__).parent.glob("*.csv"):
			print(x.stem)

	@classmethod
	def read(cls, data_name: str):
		"""This method reads the dataset specified.

    	Args:
        	data_name (str): Specifies the path of data to be read.
		
		Returns:
			pd.DataFrame: The specified dataset
    	"""
		file_path = pth(__file__).parent / pth("%s.csv"%data_name)
		if not file_path.exists():
			raise FileNotFoundError('The specified dataset does not exist. Run Dataset.list_all() to list tha available datasets.')
		else:
			X = pd.read_csv(file_path, index_col=0, header=0, na_values='NaN')
			X = X.fillna(0)
			return X
