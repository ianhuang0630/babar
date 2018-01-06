
###############################################
# Utility functions to work with Penn Tree Bank data structures
###############################################

def clean(s):

	PITAexceptions = {"-LCB-":"{", 
						"-RCB-":"}", 
						"-LRB-":"(", 
						"-RRB-":")", 
						"*LCB*":"{",
						"*RCB*":"}",
						"*LRB*":"(",
						"*RRB*":")"}

	while len(s)>0 and s[0] == "(":
		s = s[1:]

	while len(s)>0 and s[-1] == ")":
		s = s[:-1]

	if s in PITAexceptions:
		s = PITAexceptions[s]

	if "/" in s:
		s = s[:s.rindex("/")]

	return s

def tree2labels(tree, label="NP", labeling_type="IOB", rules={"O": 0, "B-NP": 1, "I-NP": 2}):
	"""
	Input:
		tree (nltk.tree.Tree): A tree with POS and NP labeling.
		label (str): label type being hunted for
		labeling_type (str): labeling type - either "IOB" or "IO"
	Returns:
		NPlabels (tuple): NP labels, according to the label type
	"""
	base_str = tree2str(tree)

	# splitting
	str_elements = base_str.split()

	np = False
	encoding = []
	first = False

	for element in str_elements:
		
		if np:
			# set either I-NP or B-NP if stack number not = 0

			# if begins with a (, then increment nesting level
			if element[0] == "(":
				## increment nesting level
				stack += 1
			else:
			# elif element[0] != "*" and element != "0":
				if first:
					## add B-NP

					encoding.append((clean(element), rules["B-NP"]))
					first = False
				else:
					## add I-NP

					encoding.append((clean(element), rules["I-NP"]))

				while len(element) > 0 and element[-1] == ")": # in case of multiple )'s
					element = element[:-1]
					## decrement nesting level
					stack -= 1 
					stack = max(stack, 0)

			# if ends with a ), then 1) add, 2) decrement nesting level

			if stack == 0:
				np = False

		elif element[:len(label)+1] == "("+label:
			## if the tag is seen
			np = True
			stack = 1
			first = True

		# if element is not an irrelevan tag
		elif element[0] != "(":
		# elif element[0] != "(" and element[0]!= "*" and element != "0":
			encoding.append((clean(element), rules["O"]))

	labels = tuple(encoding)

	return labels

def tree2str(tree):
	"""
	Input:
		tree (nltk.tree.Tree): a tree with POS and NP labeling.
	
	Returns:
		tree_string (str): Pre-order traversal of the tree
	"""

	return str(tree)


if __name__ == "__main__":

	# Testing on default tree.
	# TODO: implement this.

	pass





