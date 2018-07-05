import pandas as pd
import glob
import os

class Datasets():

	@classmethod
	def list_all(cls):
		os.chdir("./datasets")
		l = glob.glob("*.csv")
		li=[x.split('.')[0] for x in l]
		for file in li:
			print(file)
		os.chdir("..")

	@classmethod
	def read(cls, data):
		X = pd.read_csv(r'%s.csv' % (data), index_col=0, header=0, na_values='NaN')
		X = X.fillna(0)
		return X
