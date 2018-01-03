import numpy as np
import pickle
import os


# TODO: for readRaw and readParsed, also return the name of the file from where
# the data was read.


"""
PennTreeLoader is the data loader for the PennTreeBank data.
"""
class PennTreeLoader:

	def __init__(self, path, NP_tagging_type="IOB"):
		"""
		Input:
			path (String): path to the Penn Tree bank directory
		"""

		self.path = path

		
		self.raw_path = os.path.join(self.path, "raw")
		self.parsed_path = os.path.join(self.path, "parsed")
		self.tagged_path = os.path.join(self.path, "tagged")

		if NP_tagging_type == "IOB":
			self.IOB = True
			self.IO = False
		else:
			self.IO = True
			self.IOB = False

		self.raw = None
		self.parsed = None
		self.tagged = None

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
									line[i] == "?" or line[i] == "!":

									line = line[:i] + " " + line[i:]

							this_file.append(tuple(line.split()))

				all_files.append(this_file)

		raw_processed = np.array(all_files)
		self.raw = raw_processed

		return raw_processed

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

		over_all_files = []


		for file in os.listdir(self.parsed_path):

			if file != "README":
				over_all_files.append([])

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
								over_all_files[-1].extend(self.process(string))
								## reset string
								string = ""

		# TODO: implement option of returning a tree
		over_all_files = [tuple(ls) for ls in over_all_files]

		parsed_labels = np.array(over_all_files)
		self.parsed = parsed_labels

		# TODO: an assert statement to verify that every length of every tuple
		# in self.parsed is the same length as a tuple in self.raw.

		return parsed_labels

	def readPOS(self):
		"""
		Returns:
			pos_tags (np.array): each element is a file, and each element is a
				tuple containing the POS labeling of each word.
		"""

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
						all_files[-1].append(element[element.index("/")+1:])

					else:
						print(" '/' not in {} in {}".format(element, file))

				all_files[-1] = tuple(all_files[-1])

		pos_tags = np.array(all_files)
		self.tagged = pos_tags

		return pos_tags

	def process(self, string, target="NP", gram_role=False):
		"""
		*** NOTE ***
		This labels all punctuations as well. Words are labeled in whole, and
		not by character.

		Inputs:
			string (str): input string.

		Returns:
			labels (tuple): label for every word in the string.
		"""

		encoding = []
		np = False
		first = False

		for element in string.split():

			# for every element
			if np:
				# set either I-NP or B-NP if stack number not = 0

				# if begins with a (, then increment nesting level
				if element[0] == "(":
					## increment nesting level
					stack += 1
				else:
					if first:
						## add B-NP
						encoding.append(self.label_rules("B-NP"))
						first = False
					else:
						## add I-NP
						encoding.append(self.label_rules("I-NP"))

					while len(element) > 0 and element[-1] == ")": # in case of multiple )'s
						element = element[:-1]
						## decrement nesting level
						stack -= 1 
						stack = max(stack, 0)

				# if ends with a ), then 1) add, 2) decrement nesting level

				if stack == 0:
					np = False
	
			elif element[:len(target)+1] == "("+target:
				## if the tag is seen
				np = True
				stack = 1
				first = True

			# if element is not an irrelevan tag
			elif element[0] != "(":
				encoding.append(self.label_rules("O")) # Label "O"

		labels = tuple(encoding)

		return labels

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




def main():

	pennloader = PennTreeLoader("/Users/ian.huang/Documents/Projects/babar/treebank/")
	

	pennloader.readRaw()
	pennloader.readParsed()
	pennloader.readPOS()

	print("Length of list for rawdata: {}".format(pennloader.raw.size))
	print("Length of list for parseddata: {}".format(pennloader.parsed.size))
	print("length of list for POSdata: {}".format(pennloader.tagged.size))

if __name__ == "__main__":
	main()


