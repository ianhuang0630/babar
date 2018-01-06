#########################################################################
# This script focuses on training the model to recognize named entities #
#########################################################################

import nltk
import numpy as np
import pandas as pd

from utils import treemethods

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataloader import PennTreeLoader

PTB = "treebank/"

LABEL_MAP = {"O": 0, "B-NP": 1, "I-NP": 2}

class VanillaNPLearner:
	"""
	A vanilla version of NP learner. Purely using the ntlk packages
	""" 

	def __init__(self, data, NP_tagging_type="IOB"):
		"""
		Inputs:
			data (np.array): 
			NP_tagging_type (string):
		"""

		# Assuming data is penntreebank
		ptl = PennTreeLoader(data, NP_tagging_type=NP_tagging_type)

		## splitting dataset into training and testing
		
		self.all_parsed = ptl.readParsed()
		self.all_pos = ptl.readPOS()

		#checking sanity of the data
		ptl.doubleCheck()

		self.parsed_train, self.parsed_test, self.pos_train, self.pos_test \
			= train_test_split(self.all_parsed, self.all_pos, test_size=0.2)


		for idx in range(self.parsed_test.size):
			assert len(self.parsed_test[idx]) == len(self.pos_test[idx]), "failed at index = {}".format(idx)

		self.predictions = None

		self.grammar = """
				NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
					{<NNP>+}                # chunk sequences of proper nouns
		"""

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
		cp = nltk.RegexpParser(self.grammar)

		lis = []

		for (idx, test_tuple) in enumerate(self.pos_test):

			sentence = list(test_tuple)
			result = cp.parse(sentence) # predicting purely based on POS labeling

			## TODO: parse result into a list of tuples. This may be in a weird tree structure.
			# Might have to use a recursive structure, or an inbuilt method in the Tree class

			result = treemethods.tree2labels(result, labeling_type="IOB", rules=LABEL_MAP)
			## strip the result tuple of the words in each tuple element (i.e. ())

			result = tuple([label for (_, label) in result])

			assert len(result) == len(self.parsed_test[idx]), "idx = {}".format(idx)
			lis.append(result)


		self.predictions = np.array(lis)
		return self.predictions


	def evaluate(self):
		"""
		Comparing output of predict() with self.parsed_test.
			
		"""

		import ipdb; ipdb.set_trace()

		flattened_predictions = []
		flattened_gt = []

		for i in range(self.predictions.size):
			flattened_predictions.extend(self.predictions[i])
			flattened_gt.extend([label for (_, label) in self.parsed_test[i]])

		assert len(flattened_gt) == len(flattened_predictions)


		# calculate Accuracy score
		accuracy = accuracy_score(flattened_gt, flattened_predictions)

		# for testing
		print(accuracy)
		# calculate confusion matrix



		

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
	vnpl = VanillaNPLearner(PTB)
	vnpl.predict()
	vnpl.evaluate()

if __name__ == "__main__":
	main()

