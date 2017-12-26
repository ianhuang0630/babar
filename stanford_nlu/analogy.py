import sys
import numpy as np
import pandas as pd
import _pickle as cPickle
import itertools

CACHED = ""
WORD_COUNT_FILE = "vsmdata/imdb-wordword.csv"
SAVE_COUNT = "vsmdata/count_labels.pkl"
N = 100

def cos_dist(v1, v2):
	"""
	Inputs:
		v1 (np.array):
		v2 (np.array):

	returns:
		cos (float):
	"""

	assert v1.dot(v1) == 1 and v2.dot(v2) == 1, "vectors are not normalized"
	return v1.dot(v2)

def find_match(target, word_vec, word_labels, comp_func):
	"""
	Input:
		target (string): target word
		word_vec (np.array): embeddings
		word_labels (list): word names
		comp_func: comparison function (here is going to be cos_dist)
	
	Returns:
		ranks (list): each item is a tuple, containing the word and then its
		cosine distance, ranked from closest to furthest.
	"""
	target_embedding = word_vec[word_labels.index(target)]
	suggestions = []

	for word in word_labels:
		if word != target:

			word_embedding = word_vec[word_labels.index(word)]

			## find cos_dist(target, word)
			cd = cos_dist(target_embedding, word_embedding)
			suggestions.append((word,cd))

	return sorted(suggestions, key=itemgetter(1), reverse=False)


def GloVe(mat, n, word_labels, xmax = 100, alpha=0.75, iterations=100,
	learning_rate=0.05, display_progress=True):
	"""
	Input:
		mat (np.array): The count matrix
		n (int): number of dimensions in the embeddings
		word_labels (list): list of labels

	Returns:
		word_vec (np.array):

		word_labels (list):
	"""

	num_words = len(word_labels)

	## initialize random biases for word
	W = np.random.rand(num_words, n)
	W_b = np.random.rand(num_words, 1)
	## initialize random biases for context
	C = np.random.rand(num_words, n)
	C_b = np.random.rand(num_words, 1)

	indices = list(range(num_words))

	prev_epoch_error = None

	for iteration in range(iterations):

		error = 0.0

		for i, j in itertools.product(indices, indices):
			if mat[i,j] > 0.0:
				print("Iteration #{}/{}, i={}, j={}, latest error = {}, current error within epoch: {}".\
					format(iteration, iterations, i, j, prev_epoch_error, error))

				weight = (mat[i,j]/xmax) ** alpha if mat[i,j] < xmax else 1.0

				diff = np.dot(W[i], C[j]) + W_b[i] + C_b[j] - np.log(mat[i,j])
				fdiff = diff * weight

				## Calculating gradients
				W_grad = fdiff * C[j] # with respect to W[i]
				W_b_grad = fdiff
				C_grad = fdiff * W[i] # with respect to C[j]
				C_b_grad = fdiff

				## Updates:
				W[i] -= learning_rate*W_grad
				W_b[i] -= learning_rate*W_b_grad

				C[i] -= learning_rate*C_grad
				C_b[i] -= learning_rate*C_b_grad

				## error
				error += 0.5 * weight * (diff**2)

		if display_progress:
			prev_epoch_error = error
			print ("iteration {}: error {}".format(iteration, error))

	import ipdb; ipdb.set_trace()

	return (W + C, rownames)


def main(ABC, verbose=False):
	"""
	Inputs:
		ABC (list): holding strings A, B and C.
		verbose (boolean, optional): if verbose, then rankings of suggestions
			would be listed. Otherwise, only one suggestion will be made
	Returns:
		
	"""

	[A, B, C] = ABC
	
	## loading data and pickling into a file
	## if already cached, just load the file

	if not CACHED:
		## load the csv file: gigawordnyt-advmod-matrix.csv
		wc = pd.read_csv(WORD_COUNT_FILE, index_col=0)

		## splitting gam into count matrix (np.array), word_labels
		word_labels = list(wc.index)
		count = wc.values # count is numpy array

		cPickle.dump((word_labels, count), open(SAVE_COUNT, "wb"))

	else:

		(word_labels, count) = cPickle.load(open(SAVE_COUNT))

	## Use GloVe to get embeddings
	embed = GloVe(count, N, word_labels, learning_rate = 0.01)

	## calculating expected embedding of target vector

	target = (A-B) + C

	## find_match
	suggestions = find_match(target, embed, word_labels, cos_dist)

	if verbose:
		[print(suggestion) for suggestion in suggestions]

	else:
		print(suggestion[0])


if __name__ == "__main__":
	
	ABC = []
	count = 0
	# for line in sys.stdin:
	# 	ABC.append(line)
	# 	count += 1
	# 	if count == 3:
	# 		break

	ABC = ["hello" , "hi", "bye"]
	import ipdb; ipdb.set_trace()
	main(ABC)

