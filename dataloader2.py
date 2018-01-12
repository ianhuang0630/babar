import numpy as np
import pickle
import os
from utils import treemethods

# TODO: for readRaw and readParsed, also return the name of the file from where
# the data was read.


"""
PennTreeLoader is the data loader for the PennTreeBank data.
"""
DEFAULT_LABEL_MAP = {"O": 0, "B-NP": 1, "I-NP": 2}

class PennTreeLoader:

	def __init__(self, path, label_map, in_sentences=True, NP_tagging_type="IOB", verbose=False):
		"""
		Input:
			path (String): path to the Penn Tree bank directory
		"""
		self.labeling_type = NP_tagging_type
		self.label_map = label_map

		self.verbose = verbose

		self.path = path

		self.inSentences = in_sentences

		self.raw_path = os.path.join(self.path, "raw")
		self.parsed_path = os.path.join(self.path, "parsed")
		self.tagged_path = os.path.join(self.path, "tagged")

		if NP_tagging_type == "IOB":
			self.IOB = True
			self.IO = False
		else:
			self.IO = True
			self.IOB = False

		# parsing files in full
		self.raw = None
		self.parsed = None
		self.tagged = None

		# sentences
		self.raw_sents = None
		self.parsed_sents = None
		self.tagged_sents = None

		# field used in readParsed()
		self.verifiable = False


		# PITA = Pain In The Ass
		self.PITAexceptions = {"-LCB-":"{", 
								"-RCB-":"}", 
								"-LRB-":"(", 
								"-RRB-":")", 
								"*LCB*":"{",
								"*RCB*":"}",
								"*LRB*":"(",
								"*RRB*":")"}

	def readRaw(self):
		"""
		Returns:
			raw_processed (np.array): each element is a string representing a 
				training.
		"""

		all_files = []

		for file in os.listdir(self.raw_path):
			if file != "README":
				this_file = []
				with open(os.path.join(self.raw_path, file)) as f:
					for line in f:
						if line != "\n" and line != ".START \n":
							
							## for consistency with readParsed(), readRaw
							## will consider punctuations as their own word. 

							# TODO: Verify that the following punctations are
							# individualized by themselves in the parsed files
							# and use regex to make this part more elegant.
							for i in range(len(line)):
								if line[i] == "." or line[i] == "," or \
									line[i] == ":" or line[i] == ";" or \
									line[i] == "?" or line[i] == "!" or \
									line[i] == "{" or line[i] == "}" or \
									line[i] == "(" or line[i] == ")":

									line = line[:i] + " " + line[i:]

							this_file.extend(line.split())

				all_files.append(tuple(this_file))

		raw_processed = np.array(all_files)

		self.raw = raw_processed

		if self.inSentences:
			self.raw_sents = self.split2sents(raw_processed)
			return self.raw_sents
		else:
			return self.raw

	def readParsed(self, target="NP", gram_role=False, return_tree=False):
		"""
		Inputs:
			target (str, optional): the chunk type that is being identified.
	
			gram_role (bool, optional): True if you'd like the computer to 
				discriminate between different kinds of grammatical roles (e.g. 
				NP-SBJ and NP-PRD would be different.) False otherwise.

			return_tree (bool, optional): True if you'd like the program to
				return a list of trees rather than a numpy array.

		Returns: 
			If not return_tree:
				parsed_labels (np.array): each element corresponding to the 
					same index in the np.array from readRaw(), each element 
					containing the numerical labeling of different classes.

		"""

		def clean(s):

			while len(s)>0 and s[0] == "(":
				s = s[1:]

			while len(s)>0 and s[-1] == ")":
				s = s[:-1]

			if s in self.PITAexceptions:
				s = self.PITAexceptions[s]

			return s



		if not self.verifiable:
			self.readPOS()

		over_all_files = []

		file_num = 0

		for file in os.listdir(self.parsed_path):
			if file != "README":
				over_all_files.append([])
				temp = []

				## stacks to keep track of nesting layers
				with open(os.path.join(self.parsed_path, file)) as f:

					# every bracket is accompanied by a label.
					# if bracket starts with target, then chop until the
					# corresponding closing bracket is closed. First word
					# in this bracket is marked "B-NP", and all others are 
					# marked "I-NP".

					# otherwise, all words until the next open bracket are 
					# marked "O"

					stack = 0

					string = ""

					for line in f:						
						for element in line:

							if element == "(":
								stack += 1

							if stack != 0:
								string += element

							if element == ")":
								stack -= 1

							if stack == 0 and len(string) > 0:
								## process the block
								# temp.extend(self.process(string))

								temp.extend(treemethods.str2labels(string, cleaner=clean, label=target, labeling_type=self.labeling_type, rules=self.label_map))

								## reset string
								string = ""

					# verify over_all_files[-1] and self.tagged[file_num]

					finalized = []
					counter = 0
					issue = False

					# because lazy evaluation doesn't work in some python versions 
					next_eq = False

					for idx,element in enumerate(temp):

						this_word = element[0]
						tagged_word = self.tagged[file_num][counter][0]

						# because lazy evaluation doesn't work in some python versions
						if counter+1 < len(self.tagged[file_num]) and idx+1 < len(temp): 
							if temp[idx+1][0] == self.tagged[file_num][counter+1][0]:
								next_eq = True
							else:
								next_eq = False
						else:
							next_eq = False

						if this_word == tagged_word or next_eq:
							# add to finalized.
							if this_word == tagged_word:

								if issue and self.verbose:
									print("***************** ISSUE FIXED *****************")
									issue = False

								finalized.append(element)
							else:
								finalized.append((tagged_word,element[1]))
							counter += 1
						else:
							if self.verbose:
								if issue:
									print("*************** ISSUE UNRESOLVED **************")
								print("'{}' not equal to '{}' in POS.".format(this_word, tagged_word))

								if counter+1 <len(self.tagged[file_num]):
									print("'{}' not equal to '{}' in POS, in the next position.".format(temp[idx+1][0], self.tagged[file_num][counter+1][0]))
									issue = True


					## VERY questionable approach, but should be very rare.
					if counter < len(self.tagged[file_num]):
						# import ipdb; ipdb.set_trace()
						if self.verbose:
							print("There are elements in self.tagged that are not in self.parsed. Chopping self.tagged.")

						self.tagged[file_num] = tuple([self.tagged[file_num][i] for i in range(counter)])


					assert len(finalized) == len(self.tagged[file_num]), \
						"length parsed are not equal."

					over_all_files[-1] = finalized

				file_num += 1

		# TODO: implement option of returning a tree
		over_all_files = [tuple(ls) for ls in over_all_files]

		parsed_labels = np.array(over_all_files)

		self.parsed = parsed_labels

		# TODO: an assert statement to verify that every length of every tuple
		# in self.parsed is the same length as a tuple in self.raw.

		if self.inSentences:
			self.parsed_sents = self.split2sents(parsed_labels)
			return self.parsed_sents
		else:
			return self.parsed

	def readPOS(self):
		"""
		Returns:
			pos_tags (np.array): each element is a file, and each element is a
				tuple containing the POS labeling of each word.
		"""


		if self.verifiable:
			if self.inSentences:
				return self.tagged_sents
			else:
				return self.tagged

		else:
			all_files = []

			for file in os.listdir(self.tagged_path):

				if file != "README":

					all_files.append([])

					words = []

					with open(os.path.join(self.tagged_path,file)) as f:
						for line in f:
							if line != "\n":

								if line[0] == "[":
									words.extend(line[1:line.index("]")].split())

								# if line[0] == "[" and line[-2:] == "]\n":
								# 	words.extend(line[1:-2].split())

								else:
									for word in line.split():
										words.append(word)

					for element in words:

						if "/" in element:

							while "\\" in element:
								a = element.index("\\")
								element = element[:a] + element[a+1:]

							## each element of the tuple is a tuple, first element
							## containing the word, second element containing the
							## POS tag.

							upto = element.rindex("/")

							all_files[-1].append((element[:upto], element[upto+1:]))

						elif self.verbose:
							print(" '/' not in {} in {}".format(element, file))

					all_files[-1] = tuple(all_files[-1])

			pos_tags = np.array(all_files)
			self.tagged = pos_tags
			self.verifiable = True

			if self.inSentences:
				self.tagged_sents = self.split2sents(pos_tags)
				return self.tagged_sents
			else:
				return self.tagged

			return pos_tags


	def split2sents(self, file_stack):
		"""
		In charge of splitting tuples of words in a whole document into
		multiple tuples, where each tuple is a sentence. 
		"""

		all_sentences = []

		for file in file_stack:

			a_sent = [] # a buffer for sentences

			for tup in file:
				if type(tup) != tuple:
					# then we're dealing with a string for self.raw
					
					if tup == ".":
						a_sent.append(tup)
						all_sentences.append(tuple(a_sent))
						a_sent = []
					else:
						a_sent.append(tup)

				else:
					# we're dealing with a label in self.parsed or self.POS
					if tup[0] == ".":
						a_sent.append(tup)
						all_sentences.append(tuple(a_sent))
						a_sent = []
					else:
						a_sent.append(tup)

		return np.array(all_sentences)
		

	def label_rules(self, label, rules={"O": 0, "B-NP": 1, "I-NP": 2}):
		"""
		Inputs:
			label (str): Label
			rules (dict): dictionary specifying the rules
		
		"""

		if label not in rules:
			raise ValueError("Label not in rules.")
		else:
			return rules[label]

	def doubleCheck(self):
		"""
		Double checks that the same symbols were read.

		"""

		fine = True 
		
		assert self.parsed.size == self.tagged.size, \
			"parsed an tagged of different length"

		for i in range(self.parsed.size):

			A = self.parsed[i]
			B = self.tagged[i]

			for j in range(min(len(A), len(B))):

				if A[j][0] != B[j][0]:
					print("parsed and tagged at index {} are not the same. {} != {}".format(j, A[j][0], B[j][0]))
			
			if len(A) != len(B):
				raise ValueError("parsed[{}] and tagged[{}] are lengths {} and {}.".\
				format(i,i, len(A), len(B)))



	def showcontext(self, A, B, j, r=5):
		"""
		For debugging. Shows the surrounding words of index j in the tuples A 
		and B. Meant to help with differentiating where a mismatch occurs, and 
		the context under which they occur.
		
		Inputs:
			A (tuple): first tuple
			B (tuple): second tuple
			j (int): index
			r (int, optional): "size" of the context
		
		""" 
		print(" ".join([A[i][0] for i in range(max(j-r, 0), min(j+r,len(A)))]))
		print(" ".join([B[i][0] for i in range(max(j-r, 0), min(j+r,len(B)))]))


def main():

	pennloader = PennTreeLoader("/Users/ian.huang/Documents/Projects/babar/treebank/", DEFAULT_LABEL_MAP, in_sentences=True, verbose=True)
	

	# raw = pennloader.readRaw()
	parsed = pennloader.readParsed()
	pos = pennloader.readPOS()

	# print("Length of list for rawdata: {}".format(raw.size))
	print("Length of list for parseddata: {}".format(parsed.size))
	print("length of list for POSdata: {}".format(pos.size))

	import ipdb; ipdb.set_trace()

	pennloader.doubleCheck()

if __name__ == "__main__":
	main()


