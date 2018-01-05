#########################################################################
# This script focuses on training the model to recognize named entities #
#########################################################################

import nltk
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from dataloader import PennTreeLoader

PTB = "treebank/"

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

		for test_tuple in self.pos_test:

			sentence = list(test_tuple)
			result = cp.parse(sentence)

			## TODO: parse result into a list of tuples. This may be in a weird tree structure.
			# Might have to use a recursive structure, or an inbuilt method in the Tree class

			lis.append(result)

		return np.array(lis)

	def evaluate(self):
		"""
		Output:
			accuracy:
		"""

		pass
		

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

if __name__ == "__main__":
	main()

