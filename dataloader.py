import numpy as np
import pickle
import os

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

	def readRaw(self):
		"""
		Returns:
			raw_processed (np.array): each element is a string representing a 
				training.
		"""

		all_files = []
		import ipdb; ipdb.set_trace()

		for file in os.listdir(self.raw_path):
			if file != "README":
				this_file = []
				with open(os.path.join(self.raw_path, file)) as f:
					for line in f:
						if line != "\n" and line != ".START \n":
							this_file.append(line)

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

		for file in os.listdir(self.raw_path):
			if file != "README":
				## stacks to keep track of nesting layers
				stack = 0

				for line in file:
					for element in line:
						if element == "(":
							stack += 1

						if element == ")":
							stack -= 1

						#TODO: complete this

				

	def label_rules(self, rules={"O": 0, "B-NP": 1, "I-NP": 2}):
		"""
		Inputs:
			rules (dict): dictionary specifying the rules
		
		"""





def main():

	pennloader = PennTreeLoader("/Users/ian.huang/Documents/Projects/babar/treebank/")
	print(pennloader.readRaw())

if __name__ == "__main__":
	main()


