
# graph of obj's, connected by relation objs.

class relation:
	def __init__(self, name):
		self.name = name
		self.transitive = False
		self.reflexive = False
		self.symmetric = False

	def isSymmetric(self):
		self.symmetric = True

	def isReflexive(self):
		self.reflexive = True

	def isTransitive(self):
		self.transitive = True
	
class obj:
	def __init__ (self, name):
		self.name = name
		self.properties = set([])
		self.adj_list = dict()

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

		if relation.transitive:
			## update adjacency list recursively
			pass

		if relation.symmetric:
			## update adjacency list - edge is non-directional
			if self not in next_obj.adj_list:
				next_obj.addRelation(relation, self)

		if relation.reflexive:
			## update adjacency list
			if self not in self.adj_list:
				self.addRelation(relation, self)

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name


def main():	

	dog = obj("dog") 
	house = obj("house")

	rel_in = relation("in")
	rel_in.isTransitive()

	dog.addProperty("black")
	dog.addRelation(rel_in, house)


	print(dog.properties)
	print(dog.adj_list)


if __name__ == "__main__":
	main()


