#########################################################################
# This script focuses on training the model to recognize named entities #
#########################################################################

from sklearn import *
import nltk
import numpy as np
import pandas as pd


POS_PENNBANK = "treebank/tagged"
NP_PENNBANK = "treebank/combined"

class VanillaNPLearner:
	"""
	A vanilla version of NP learner. Purely using the ntlk packages
	""" 

	def __init__(self, data):
		"""
		Inputs:
			data (np.array): 
		"""

		## splitting dataset into training and testing

		pass


	def fit(self):
		pass # for consistencey with NPLearner

	def get_features(self):
		pass # for consistency with NPLearner

	def predict(self):
		"""
		Input:
			POS tags
		Output:
			NP labels
		"""


	def evaluate(self):
		"""

		"""

class NPLearner:
	def __init__(self):
		pass

	def fit(self):
		pass

	def get_features(self):
		pass

	def predict(self):
		pass


def main():

	# getting data into list of lists of tuples.
	
	# for every file

		with open(POS_PENNBANK) as f:
			for line in f:
				




def clean_data(dataset):



if __name__ == "__main__":
	main()

