from collections import deque
# graph of obj's, connected by relation objs.

class Relation:
	def __init__(self, name):
		self.name = name
		self.transitive = False
		self.reflexive = False
		self.symmetric = False

		self.inverse_exists = False
		self.inverse = None

	def isSymmetric(self):
		self.symmetric = True

	def isReflexive(self):
		self.reflexive = True

	def isTransitive(self):
		self.transitive = True
	
	def hasInverse(self, other_relation):
		self.inverse_exists = True
		self.inverse = other_relation

class Obj:
	def __init__ (self, name):
		self.name = name
		self.properties = set([])
		self.adj_list = dict()

		## for graph traversal
		self.prev = None
		self.known = False

	def addProperty(self, prop):
		if type(prop) == list:
			for p in prop:
				self.properties.add(p)
		if type(prop) == str:
			self.properties.add(prop)
		else:
			raise ValueError("Property has to be string or list of strings")


	def addRelation(self, relation, next_obj):

		## adding to the adjacency list
		if relation.name not in self.adj_list:
			self.adj_list[relation.name] = [next_obj]
		else:
			self.adj_list[relation.name].append(next_obj)

		## TODO: adding the inverse of relation, if there is one
		if relation.inverse_exists and relation.inverse.name not in next_obj.adj_list:
			next_obj.addRelation(relation.inverse, self)


		## dealing with relation properties

		if relation.transitive:
			## update adjacency list recursively
			pass

		if relation.symmetric:
			## update adjacency list - edge is non-directional
			if self not in next_obj.adj_list[relation.name]:
				next_obj.addRelation(relation, self)

			# question: if R is symmetric, then is R^-1 also symmetric

		if relation.reflexive:
			## update adjacency list
			if self not in self.adj_list[relation.name]:
				self.addRelation(relation, self)

			# question: if R is reflexive, then is R^-1 also reflexive

	def reset(self):
		self.prev = None
		self.known = False

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name


class Question:
	def __init__(self, origin, target):
		# Ideally the input should be a string, and the class question would use
		# NL parsing to extract features of the question

		"""
		Input: 
			origin (Obj): Baseline object
			target (Obj): target object

		"""

		# describe the location of the target in relation to the origin
		
		self.origin = origin
		self.target = target

		# we could do some fancy NLP parsing in this class!

	def get_path(self):
		"""
		Do a breadth first search through graph until the relation to the 
		target object is found
		"""

		thisNode = self.target

		d = deque()
		d.append(thisNode)

		while d:
			next_obj = d.popleft()
			next_obj.known = True

			if next_obj.name == self.origin.name:
				break

			# go through every adjacent node, and add (relation, next_obj)
			for relation in next_obj.adj_list:
				for node in next_obj.adj_list[relation]:

					if not node.known:
						node.prev = (relation, next_obj)
						d.append(node)

		## end of BFS.

		trans_el_stack = []

		while next_obj != None:
			# add to the front of the list next_obj.prev[1]
			# add next_obj.prev[0]

			if next_obj.prev != None:
				trans_el_stack.append(next_obj.prev[0])
				trans_el_stack.append(next_obj.prev[1])
				next_obj = next_obj.prev[1]
			else:
				next_obj = None

		## inverting the stack

		trans_el = []

		while trans_el_stack:
			trans_el.append(trans_el_stack.pop())

		return trans_el

	def form_answer(self):
		"""
		Vanilla implementation of how the computer would give an answer
		
		We could totally replace this part with some fancy natural language 
		generation, with random probabilities of describing the objects under
		question. (e.g. the *white* cat is under the tree...)

		"""

		trans_obj = self.get_path()

		answer = ""
		noun_count = 0

		for element in trans_obj:

			if type(element) is Obj:
				answer += "the " + element.name 

				noun_count += 1

				if noun_count < 2:
					answer += " is "
				else:
					answer += ", which is "

			if type(element) is str:
				answer += element + " "


		answer += "the " + self.origin.name + "."

		return answer



def main():	

	dog = Obj("dog") 
	house = Obj("house")

	rel_in = Relation("in")
	rel_in.isTransitive()

	rel_outside = Relation("outside")
	rel_outside.isTransitive()

	rel_in.hasInverse(rel_outside)
	rel_outside.hasInverse(rel_in)


	dog.addProperty("black")
	dog.addRelation(rel_in, house)

	cat = Obj("cat")

	rel_on = Relation("on")
	rel_on.isTransitive()

	rel_under = Relation("under")
	rel_under.isTransitive()

	rel_on.hasInverse(rel_under)
	rel_under.hasInverse(rel_on)

	cat.addProperty("white")
	cat.addRelation(rel_on, house)

	print(dog.properties)
	print(cat.properties)
	print(house.properties)

	print("House adj list: {}".format(house.adj_list))
	print("Dog adj list: {}".format(dog.adj_list))
	print("Cat adj list: {}".format(cat.adj_list))

	## question: where is the dog in relation to the cat?
	## this requires some relations (e.g. on, in ...etc) to be invertible

	## expected answer: the dog is in the house, which is under the cat

	question = Question(cat, dog)

	print("----------------------------------------------------")

	print("Question: Where is the dog in relation to the cat?")
	print(question.form_answer())

	dog.reset()
	cat.reset()
	house.reset()

	print("\n ")
	print("Question: Where is the cat in relation to the dog?")

	question2 = Question(dog, cat)
	print(question2.form_answer())


if __name__ == "__main__":
	main()


