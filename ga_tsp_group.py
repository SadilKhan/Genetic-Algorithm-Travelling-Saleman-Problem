import Reporter
import numpy as np
import matplotlib.pyplot as plt

# Modify the class name to match your student number.
class r0873969:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename,k=5):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Initialize necessary variables
		self.population = self.initialize_population()
		self.distanceMat = distanceMatrix
		self.generation=10000 # Number of Iteration
		self.recombProb = 1
		self.mutationProb = 0.01
		self.k=k
		i=0

		self.lambdaa = len(self.population) # population Size
		self.meanSolutions=[] # mean of the fitness score per generations
		self.bestSolutions=[] # Value of the best solutions per generations
		self.solutions=[]
		

		# Your code here.
		while(True):
			# Selection
			selected=self.selection()
			# Crossover
			offspring=self.crossover(selected)   
			# Mutation  
			mutatedOffspring=self.mutation(offspring)
			# Addition of offspring in the population
			newPopulation=np.concatenate((self.population,mutatedOffspring),axis=0)

			#Eliminate the worst performing candidates from the population
			self.elimination(newPopulation)
			print(f"Generation {i} The best solution is {self.bestSolutions[-1]} and route is {self.solutions[-1]}")

			# Get the last scores
			meanObjective=self.meanSolutions[-1]
			bestObjective=self.meanSolutions[-1]
			bestSolution=self.solutions[-1]
			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break
			i+=1

		# Your code here.
		return 0

	def crossover(self,selected):
		""" Ordered crossover"""
		offspring=[]
		for i in range((len(selected)//2)):
			child=[]
			childP1=[]
			childP2=[]
			random_cut=np.sort(np.random.choice(range(1,29),2))
			parent1=selected[2*i]
			parent2=selected[2*i+1]
			childP1+=list(parent1[random_cut[0]:random_cut[1]])
			childP2=[chr for chr in parent2 if chr not in childP1]
			child=childP2[:random_cut[0]]+childP1+childP2[random_cut[0]:]
			offspring.append(child)
		print(len(offspring))
		return np.array(offspring)

	def fitness(self,population):
		"""Calculate the total distance for the route"""

		scorePop=[]
		for gene in population:
			score=0
			#print(self.distanceMat.shape)
			for i in range(len(gene)):
				score+=self.distanceMat[gene[i],gene[(i+1)%len(gene)]]
			scorePop.append(score)
		return scorePop
	
	def selection(self):
		"""K tournament Selection"""
		selectedSize=int(self.lambdaa//2)

		selected=[]
		for i in range(selectedSize):
			# Choose k random candidates for the tournament
			choices=self.population[np.random.choice(range(self.lambdaa),self.k)]#????

			# Evalute fitness for all candidates in the tournament
			scores=self.fitness(choices)

			# Choose the best of them
			bestCandidate=choices[np.argsort(scores)[0]]
			selected.append(bestCandidate)
		return np.array(selected)
	
	def mutation(self,offspring):
		""" Inversion Mutation """

		for i in range(len(offspring)):
			if np.random.rand()<=self.mutationProb:
				inversePos=np.random.randint(1,29,2)
				sublist=offspring[i][inversePos[0]:inversePos[1]]
				sublist= np.flip(sublist)
				offspring[i][inversePos[0]:inversePos[1]]=sublist
		mutatedOffspring=offspring
		return mutatedOffspring

	
	def elimination(self,newPopulation):
		""" Take only top 100 scoring candidates"""
		fitnessScorePop=self.fitness(newPopulation)
		sortScorePop=np.argsort(fitnessScorePop)
		self.bestSolutions.append(np.min(fitnessScorePop))
		self.solutions.append(newPopulation[np.argmin(fitnessScorePop)])
		self.meanSolutions.append(np.mean(fitnessScorePop))
		self.population=newPopulation[sortScorePop[:100]]

	def initialize_population(self,numPopulation=100):
		"""Initialize The Poplulation"""
		population=[]

		for i in range(numPopulation):
			population.append(self.permutation())
		
		population=np.array(population)
		return population

	def permutation(self):
		""" Returns a permuation of C Integers representing a list of cities."""
		gene=np.array([0])
		gene=np.append(gene,np.random.permutation(list(range(1,29))))
		return gene
	
	def convergenceTest(self):
		""" Check if the solutions improved in the last 100 solutions"""
		if len(self.bestSolutions)<5000:
			return True
		
		else:
			if np.mean(self.bestSolutions[-5000:])==self.bestSolutions[-1]:
				return True
			else:
				return False



if __name__=="__main__":
	fn=r0873969()
	fn.optimize("tour29.csv",5)
	#plt.plot(range(fn.generation),fn.bestSolutions[1:])
	plt.show()
