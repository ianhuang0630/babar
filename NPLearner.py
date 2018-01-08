#########################################################################
# This script focuses on training the model to recognize named entities #
#########################################################################

import nltk
from nltk.classify import MaxentClassifier, ConditionalExponentialClassifier,DecisionTreeClassifier, NaiveBayesClassifier, WekaClassifier
import numpy as np
import pandas as pd

from utils import treemethods

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from dataloader2 import PennTreeLoader

PTB = "treebank/"

IOB_LABEL_MAP = {"O": 0, "B-NP": 1, "I-NP": 2}
IO_LABEL_MAP = {"O": 0, "I-NP": 1}
CLASSIFIER_MAP = {
	"Maxent": MaxentClassifier, 
	"ConditionalExp": ConditionalExponentialClassifier,
	"DecisionTree": DecisionTreeClassifier,
	"NaiveBayes": NaiveBayesClassifier,
	"Weka": WekaClassifier
	}

class VanillaNPLearner:
	"""
	A vanilla version of NP learner. Purely using the ntlk packages
	""" 

	def __init__(self, data, label_map = IOB_LABEL_MAP, NP_tagging_type="IOB"):
		"""
		Inputs:
			data (np.array): 
			NP_tagging_type (string):
		"""
		self.labeling_type = NP_tagging_type
		self.label_map = label_map

		# Assuming data is penntreebank
		ptl = PennTreeLoader(data, label_map=self.label_map, NP_tagging_type=self.labeling_type)

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
			parsed_sentence = cp.parse(sentence) # predicting purely based on POS labeling

			result = treemethods.tree2labels(parsed_sentence, labeling_type=self.labeling_type, rules=self.label_map)
			## strip the result tuple of the words in each tuple element

			# ################# For Debugging ############################
			# for i in range(min(len(result), len(self.parsed_test[idx]))):

			# 	# compare result[i][0] to self.parsed_test[idx][i][0]

			# 	if result[i][0] != self.parsed_test[idx][i][0]:
			# 		print("{} != {}".format(result[i][0], self.parsed_test[idx][i][0])) 
			# 		print (sentence[i][0]) 
			# 		raise ValueError

			# ############################################################

			result = tuple([label for (_, label) in result])
			assert len(result) == len(self.parsed_test[idx]), "idx = {}".format(idx)
			
			lis.append(result)


		self.predictions = np.array(lis)
		return self.predictions


	def evaluate(self):
		"""
		Comparing output of predict() with self.parsed_test.
			
		"""

		flattened_predictions = []
		flattened_gt = []

		for i in range(self.predictions.size):
			flattened_predictions.extend(self.predictions[i])
			flattened_gt.extend([label for (_, label) in self.parsed_test[i]])

		assert len(flattened_gt) == len(flattened_predictions)


		# calculate Accuracy score
		accuracy = accuracy_score(flattened_gt, flattened_predictions)

		# for testing
		print("\n")
		print("Accuracy \n-----------------")
		print(accuracy)

		# calculate confusion matrix
		cm = confusion_matrix(flattened_gt, flattened_predictions)

		print("\n")
		print("Confusion Matrix \n------------------")
		print(cm)

		return accuracy, cm

class NPLearner:
	def __init__(self, data, label_map = IOB_LABEL_MAP, NP_tagging_type="IOB", classifier = "Maxent"):
		"""
		Inputs:
			data (np.array): 
			NP_tagging_type (string):
		"""
		self.labeling_type = NP_tagging_type
		self.label_map = label_map

		self.classifier = CLASSIFIER_MAP[classifier]

		# Assuming data is penntreebank
		ptl = PennTreeLoader(data, label_map=self.label_map, NP_tagging_type=self.labeling_type)

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


	def fit(self):
		pass

	def get_features(self):
		pass

	def predict(self):
		pass

	def evaluate(self):
		pass


def main():

	print("IOB labeling for NP using hard-coded rules...")
	vnpl = VanillaNPLearner(PTB)
	vnpl.predict()
	vnpl.evaluate()
		
	print("\n")
	print("IO labeling for NP using hard-coded rules...")
	io_vnpl = VanillaNPLearner(PTB, NP_tagging_type = "IO")
	io_vnpl.predict()
	io_vnpl.evaluate()
if __name__ == "__main__":
	main()

