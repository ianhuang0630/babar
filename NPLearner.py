#########################################################################
# This script focuses on training the model to recognize named entities #
#########################################################################

import nltk

	
from nltk.classify import SklearnClassifier
from nltk.classify import MaxentClassifier, \
						ConditionalExponentialClassifier,\
						DecisionTreeClassifier, \
						NaiveBayesClassifier, \
						WekaClassifier
import numpy as np
import pandas as pd

from utils import treemethods

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from dataloader2 import PennTreeLoader

PTB = "treebank/"

IOB_LABEL_MAP = {"O": 0, "B-NP": 1, "I-NP": 2}
IO_LABEL_MAP = {"O": 0, "I-NP": 1}

class VanillaNPLearner:
	"""
	A vanilla version of NP learner. Purely using the ntlk packages
	""" 

	def __init__(self, data, label_map = IOB_LABEL_MAP, NP_tagging_type="IOB"):
		
		self.labeling_type = NP_tagging_type
		self.label_map = label_map

		# Assuming data is penntreebank
		ptl = PennTreeLoader(data, label_map=self.label_map, 
							NP_tagging_type=self.labeling_type)

		## splitting dataset into training and testing
		
		self.all_parsed = ptl.readParsed()
		self.all_pos = ptl.readPOS()
		import ipdb; ipdb.set_trace()

		#checking sanity of the data
		ptl.doubleCheck()

		self.parsed_train, self.parsed_test, self.pos_train, self.pos_test \
			= train_test_split(self.all_parsed, self.all_pos, test_size=0.2)


		for idx in range(self.parsed_test.size):
			assert len(self.parsed_test[idx]) == len(self.pos_test[idx]), \
				"failed at index = {}".format(idx)

		
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
			# predicting purely based on POS labeling
			parsed_sentence = cp.parse(sentence) 

			result = treemethods.tree2labels(parsed_sentence, l
						abeling_type=self.labeling_type, rules=self.label_map)
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

			assert len(result) == len(self.parsed_test[idx]), \
					"idx = {}".format(idx)
			
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
	def __init__(self, data_path, model, feat_func,
				label_map = IOB_LABEL_MAP, NP_tagging_type="IOB", verbose=False):
		
		"""
		For experimenting with different models and feature functions

		Input:
			data_path (str): path to the penntreebank
			model (Model): input model
			feat_func (func): feature function
			label_map (dict, optional): mapping from NP labeling to the integer
				labels. (e.g. {"I-NP": 1, "O":0})
			NP_tagging_type (str, optional): Either "IOB" or "IO"

		"""
		
		# model and feature functions being tested.
		self.model = model
		self.feat_func = feat_func

		self.labeling_type = NP_tagging_type
		self.label_map = label_map
		self.verbose = verbose

		## Assuming data is penntreebank
		ptl = PennTreeLoader(data_path, label_map=self.label_map, \
							NP_tagging_type=self.labeling_type)
		self.all_parsed = ptl.readParsed()
		self.all_pos = ptl.readPOS()

		## checking sanity of the data
		ptl.doubleCheck()


		## calculating features
		X = self.feat_func(self.all_pos)
		y = [label for (_, label) in sent for sent in self.parsed_train]
		## 

		self.X_train, self.X_test, self.y_train, self.y_test = \
			train_test_split(X,y, test_size=0.2)

		self.predictions = None

	def fit(self):
		"""
		Fit to the dataset
		"""
		## extract features
		X = self.X_train
		y = self.y_train

		train_data = list(zip(X,y)) # in Python 3.6, the list() cast is necessary

		## train_data is of of the format:
			#[({"a": 4, "b": 1, "c": 0}, "ham"),
			# ({"a": 5, "b": 2, "c": 1}, "ham"),
			# ({"a": 0, "b": 3, "c": 4}, "spam"),
			# ({"a": 5, "b": 1, "c": 1}, "ham"),
			# ({"a": 1, "b": 4, "c": 3}, "spam")]

		self.model.fit(train_data)

	def predict(self):
		"""
		Predict based on test set
		"""

		X = self.X_test
		self.predictions = self.model.predict(test_data)

	def evaluate(self):
		"""
		Evaluating comparison between self.y_train and self.predictions
		"""	

		ac = accuracy_score(self.y_train, self.predictions)
		cm = confusion_matrix(self.y_train, self.predictions)

		# TODO: calculate F1 score if self.labeling_type == "IO"

		if self.verbose:

			# Display accuracy score
			print("\n")
			print("Accuracy \n-----------------")
			print(ac)

			# Display confusion matrix
			print("\n")
			print("Confusion Matrix \n------------------")
			print(cm)

		return ac, cm

	def getModel(self):
		return self.model 


class Model:
	def __init__(self):
		pass

	def fit(self):
		pass

	def predict(self):
		pass


def feature_func(sents):
	"""
	Input:
		sents (np.array): Each row is a tuple with the correct POS labeling
	
	Output:
		feats (list): each element is a dictionary. Dictionaries contain the
			name of the feature as the values and the values of the features
			as dictionary values. 

			e.g.[{"a": 3, "b": 2, "c": 1},
				 {"a": 0, "b": 3, "c": 7}]
	"""

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

