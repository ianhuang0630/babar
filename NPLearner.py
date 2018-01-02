#########################################################################
# This script focuses on training the model to recognize named entities #
#########################################################################

from sklearn import *
import nltk
import numpy as np
import pandas as pd

TO_NER = "entity-annotated-corpus/ner_dataset.csv"

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
	dataset = pd.read_csv(open(TO_NER, mode="r", encoding="ascii", errors="ignore"))
	

