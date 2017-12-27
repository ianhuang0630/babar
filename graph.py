
# graph of obj's, connected by relation objs.

class relation:
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

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name



def main():	

	dog = obj("dog") 
	house = obj("house")

	rel_in = relation("in")
	rel_in.isTransitive()

	rel_outside = relation("outside")
	rel_outside.isTransitive()

	rel_in.hasInverse(rel_outside)
	rel_outside.hasInverse(rel_in)


	dog.addProperty("black")
	dog.addRelation(rel_in, house)

	cat = obj("cat")

	rel_on = relation("on")
	rel_on.isTransitive()

	rel_under = relation("under")
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

	## question: where is the cat in relation to the dog?
	## this requires some relations (e.g. on, in ...etc) to be invertible

if __name__ == "__main__":
	main()


