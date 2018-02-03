import numpy as np
import _pickle as cPickle
import os
from operator import itemgetter
from distfuncs import *

GLOVE_FILE = "stanford_nlu/glove.6B/glove.6B.50d.txt"
CACHE_ADDRESS = GLOVE_FILE[:-4] + ".pkl"

class Glove:
	def __init__(self, dist_func=cosine):
		if os.path.exists(CACHE_ADDRESS):
			self.word2vec = cPickle.load(open(CACHE_ADDRESS, "rb"))
		else:
			self.word2vec = self.glove2dic()
			cPickle.dump(self.word2vec, open(CACHE_ADDRESS, "wb"))

		self.dist_func = dist_func

	def findNNeighbors(self, N, word, dist_func=None):
		"""
		Inputs:
			N (int): the number of input that one would like returned
			dist_func (func): the distance function
		Returns:
			suggestions (list): every element is a tuple, with the first
				element being the word, and the second being the distance
		"""

		if dist_func == None:
			dist_func = self.dist_func

		if word not in self.word2vec:
			raise ValueError("Word '{}' is not in dataset".format(word))
		else:
			word_vec = self.word2vec[word]

		word_dist = []

		for other in self.word2vec:
			if other != word:
				other_vec = self.word2vec[other]
				word_dist.append((other, dist_func(word_vec, other_vec)))

		return sorted(word_dist, key=itemgetter(1), reverse=False)[:N]

	
	def getVec(self, word):
		"""
		Inputs:
			word (str): input word
		"""
		return self.word2vec[word]

	def areSimilar(self, words, threshold, dist_func=None):
		## TODO: come up with threshold

		"""
		Inputs:
			words (list): list of strings, each string is a words.
			dist_func (function): distance function.
			threshold (float): distance threshold for deeming two words 
				semantically similar.

		Returns:
			similar (boolean): True if the difference of the words in the list 
				(as measured by dist_func) is not over the threshold.
		"""
	
		if dist_func == None:
			dist_func = self.dist_func

		for word1 in words:
			for word2 in words:
				
				if word1 not in self.word2vec:
					raise ValueError("Word '{}' is not in the known dataset.".\
						format(word1))
				if word2 not in self.word2vec:
					raise ValueError("Word '{}' is not in the known dataset.".\
						format(word2))

				embed1 = self.word2vec[word1]
				embed2 = self.word2vec[word2]

				if dist_func(embed1, embed2) > threshold:
					return False

		return True


	def glove2dic(self):

		""" Returns a dictionary where the keys corresponds to words and the 
		values are their embeddings.
		
		"""

		def checkLength(u):
			return abs(1-u.dot(u)) < 0.00001

		with open(GLOVE_FILE) as f:
			content = f.readlines()

		content = [x.strip() for x in content]

		dic = {}

		for string in content:
			l = string.split()
			word = l[0]
			vector = np.array([float(element) for element in l[1:]])
			vector = normalize(vector)

			assert checkLength(vector), "this vector is not normalized"

			dic[word] = vector


		return dic
	


def main():
	glove = Glove()
	print(glove.findNNeighbors(10, "happy"))

if __name__ == "__main__":

	main()
