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

			result = treemethods.tree2labels(parsed_sentence,
						labeling_type=self.labeling_type, rules=self.label_map)
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
	def __init__(self, data_path, model_types, feat_func,
				label_map = IOB_LABEL_MAP, NP_tagging_type="IOB", 
				verbose=False, max_iter=100):
		
		"""
		For experimenting with different models and feature functions
		Input:
			data_path (str): path to the penntreebank
			model_types (list): list of class pointers
			feat_func (func): feature function
			label_map (dict, optional): mapping from NP labeling to the integer
				labels. (e.g. {"I-NP": 1, "O":0})
			NP_tagging_type (str, optional): Either "IOB" or "IO"
		"""

		
		# model and feature functions being tested.
		self.model_types = model_types
		self.num_models = len(self.model_types)

		self.model = [None] * self.num_models
	
		if type(max_iter) == int:
			self.max_iter = [max_iter] * self.num_models
		else:
			self.max_iter = max_iter

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
		y = [label for sent in self.all_parsed for (_, label) in sent]

		self.X_train, self.X_test, self.y_train, self.y_test = \
			train_test_split(X,y, test_size=0.2)

		self.predictions = [None] * self.num_models

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

		for i, mod in enumerate(self.model_types):
			self.model[i] = mod.train(train_data, max_iter=self.max_iter[i])

	def predict(self):
		"""
		Predict based on test set
		"""
		X = self.X_test

		for i,mod in enumerate(self.model):
			self.predictions[i] = mod.classify_many(X)

	def evaluate(self):
		"""
		Evaluating comparison between self.y_train and self.predictions
		"""	

		metrics = []

		for i, pred in enumerate(self.predictions):
			ac = accuracy_score(self.y_train, pred)
			cm = confusion_matrix(self.y_train, pred)
			
			## CONDER: calculate F1 score if self.labeling_type == "IO"

			if self.verbose:
				print("For model: {}".format(str(self.model_types[i])))
				# Display accuracy score
				print("\n")
				print("Accuracy \n-----------------")
				print(ac)
				# Display confusion matrix
				print("\n")
				print("Confusion Matrix \n------------------")
				print(cm)

			metrics.append({"Model type": str(self.model_types[i]), "Accuracy score": ac, "confusion_matrix": cm})

		return metrics

	def getModels(self):
		"""
		Returns the list of models trained
		"""
		return self.model 


"""
Wrapper class for the classifier -- unused currently. May be more useful
when we are building Neural Networks to do similar jobs
"""
class Model:
	def __init__(self, model):
		self.model = model

	def fit(self, train_data):
		self.model.train(train_data)

	def predict(self, test_data):
		return self.model.classify_many(test_data)

"""
template feature function
"""
def default_feature_func(sents):
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
	feats = []

	for tagged_sent in sents:
		history = []
		for i, (word, tag) in enumerate(tagged_sent):
			featureset = npchunk_features(tagged_sent, i, history)
			feats.append(featureset)
			history.append(tag)

	return feats



"""
Copied from 3.3 in book
"""
def npchunk_features(sentence, i, history):
    '''
    Extracting various features to improve chunker performance
    '''
    word, pos = sentence[i]
    if i == 0:
         prevword, prevpos = "<START>", "<START>"
    else:
         prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
         nextword, nextpos = "<END>", "<END>"
    else:
         nextword, nextpos = sentence[i+1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevpos+pos": "%s+%s" % (prevpos, pos), 
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)} 
"""
Copied from 3.3 in book
"""
def tags_since_dt(sentence, i):
    '''
    Output:
        String that describes the set of all part-of-speech tags encountered since beginning of sentence OR most recent determiner 'DT'
    '''
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))


def main():

	# print("IOB labeling for NP using hard-coded rules...")
	# vnpl = VanillaNPLearner(PTB)
	# vnpl.predict()
	# vnpl.evaluate()
		
	# print("\n")
	# print("IO labeling for NP using hard-coded rules...")
	# io_vnpl = VanillaNPLearner(PTB, NP_tagging_type = "IO")
	# io_vnpl.predict()
	# io_vnpl.evaluate()

	mods = [MaxentClassifier]

	import ipdb; ipdb.set_trace()

	## setting max_iter to be 10 so that we have a proof of concept.
	npl = NPLearner(PTB, mods, default_feature_func, verbose=True, max_iter=100)

	npl.fit()
	npl.predict()
	accuracy, confusion_matrix = npl.evaluate()


	# TODO: run on all other NLTK classifiers and some scikitlearn classifiers
	# (using SklearnClassifier)


if __name__ == "__main__":
	main()