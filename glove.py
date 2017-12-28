import numpy as np
import _pickle as cPickle
import os

GLOVE_FILE = "stanford_nlu/glove.6B/glove.6B.50d.txt"
CACHE_ADDRESS = GLOVE_FILE[:-4] + ".pkl"

class Glove:
	def __init__(self):
		if os.path.exists(CACHE_ADDRESS):
			self.word2vec = cPickle.load(open(CACHE_ADDRESS, "rb"))
		else:
			self.word2vec = self.glove2dic()
			cPickle.dump(self.word2vec, open(CACHE_ADDRESS, "wb"))


	def findNNeighbors(self):
		pass

	def areSimilar(self, words, dist_func, threshold):
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

		with open(GLOVE_FILE) as f:
			content = f.readlines()

		content = [x.strip() for x in content]

		dic = {}

		for string in content:
			l = string.split()
			word = l[0]
			vector = np.array([float(element) for element in l[1:]])
			dic[word] = vector


		return dic


def main():
	glove = Glove()

if __name__ == "__main__":

	main()
