import Reporter
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import PMX,install_package
try:
	from joblib import Parallel,delayed
except:
	install_package('joblib')
# Modify the class name to match your student number.
class r0873969:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

		# Best Solution yet in tour750: 117388
		# Best Solution in tour250: 45789

	# The evolutionary algorithm's main loop
	def optimize(self, filename,k=5):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Initialize necessary variables
		self.distanceMat = distanceMatrix
		self.noinfmax=self.distanceMat[self.distanceMat!=np.inf].max()*1000
		self.distanceMat=np.nan_to_num(self.distanceMat,copy=True,posinf=self.noinfmax)
		self.numCity=len(self.distanceMat)
		self.population = self.initialize_population(200)
		self.generation=30000 # Number of Iteration
		self.recombProb = 1
		self.mutationProb = 0.4
		self.localSearchProb=0.01
		self.k=k
		self.alpha=0.99
		i=0

		self.lambdaa = len(self.population) # population Size
		self.meanSolutions=[] # mean of the fitness score per generations
		self.bestSolutions=[] # Value of the best solutions per generations
		self.solutions=[]
		

		# Your code here.
		while(True):
			# Selection
			selected=self.selection(self.population,i+1,int(self.lambdaa//2))
			# Crossover
			offspring=self.crossover1(selected)   
			# Mutation  
			mutatedOffspring=self.mutation_inverse(offspring)
			# Addition of offspring in the population
			#newPopulation=np.concatenate((self.population,mutatedOffspring),axis=0)

			#Eliminate the worst performing candidates from the population
			self.elimination(self.population,mutatedOffspring)
			#print(f"Generation: {i}, Best Solution:{self.bestSolutions[-1]}, Route: {self.solutions[-1]}")
			#print(f"Generation: {i}, Best Solution:{self.bestSolutions[-1]}")

			# Get the last scores
			meanObjective=self.meanSolutions[-1]
			bestObjective=self.bestSolutions[-1]
			bestSolution=self.solutions[-1]
			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			print(f"Generation: {i}, Best Solution:{self.bestSolutions[-1]}, TimeLeft:{timeLeft}")
			if timeLeft < 0:
				break
			i+=1

		# Your code here.
		return 0

	def crossover1(self,selected):
		""" Ordered crossover"""
		offspring=[]
		selectedSize=len(selected)
		bestCandCrossoverSize=int(0.3*selectedSize)
		for i in range(bestCandCrossoverSize):
			parent1=selected[2*i]
			parent2=selected[2*i+1]
			for j in range(2):
				child=[]
				childP1=[]
				childP2=[]
				random_cut=np.sort(np.random.choice(range(1,self.numCity),2))
				if j==1:
					parent1,parent2=parent2,parent1
				childP1+=list(parent1[random_cut[0]:random_cut[1]])
				childP2=[chr for chr in parent2 if chr not in childP1]
				child=childP2[:random_cut[0]]+childP1+childP2[random_cut[0]:]
				if np.random.rand()<=self.localSearchProb:
					child=self.k_opt(child)
				offspring.append(child)
		for i in range(selectedSize-2*bestCandCrossoverSize):
			x1,x2=np.random.permutation(list(range(selectedSize)))[:2]
			parent1=selected[x1]
			parent2=selected[x2]
			child=[]
			childP1=[]
			childP2=[]
			random_cut=np.sort(np.random.choice(range(1,self.numCity),2))
			childP1+=list(parent1[random_cut[0]:random_cut[1]])
			childP2=[chr for chr in parent2 if chr not in childP1]
			child=childP2[:random_cut[0]]+childP1+childP2[random_cut[0]:]
			if np.random.rand()<=self.localSearchProb:
				child=self.k_opt(child)
			offspring.append(child)
		return np.array(offspring)

	def fitness(self,population):
		"""Calculate the total distance for the route"""

		scorePop=[]
		for gene in population:
			try:
				score=self.distanceMat[gene,np.roll(gene,-1)].sum()
			except:
				print(gene)
				break
			scorePop.append(score)
		return scorePop
	
	def findIndex(self,array,value):
		"""Returns the last index of the array that is smaller than value"""

		for i in range(len(array)):
			if (array[i]>value) & (i==0):
				return 0
			elif array[i]>value:
				return i-1


	
	def crossover(self,selected):
		"""Partially Mapped Crossover"""
		offspring=[]
		selectedSize=len(selected)
		bestCandCrossoverSize=int(0.3*selectedSize)
		allChilds=Parallel(n_jobs=2)(delayed(PMX)(selected[2*i],selected[2*i+1]) for i in range(bestCandCrossoverSize))
		for c in allChilds:
			child1,child2=c
			if np.random.rand()<=self.localSearchProb:
				child1=self.k_opt(child1)
				child2=self.k_opt(child2)
			offspring.append(child1)
			offspring.append(child2)
		parentIndex=np.random.randint(1,selectedSize,(selectedSize-bestCandCrossoverSize,2))
		allChilds=Parallel(n_jobs=2)(delayed(PMX)(selected[i[0]],selected[i[1]]) for i in parentIndex)
		for c in allChilds:
			child1,child2=c
			if np.random.rand()<=self.localSearchProb:
				child1=self.k_opt(child1)
				child2=self.k_opt(child2)
			offspring.append(child1)
			offspring.append(child2)
		return np.array(offspring)
	

	def selection(self,population,iteration,selectedSize):
		"""Rank Selection"""
		selected=[]
		# Calculate Fitness Scores
		fitnessScorePop=self.fitness(self.population)
		fitnessRankPop=np.argsort(fitnessScorePop)
		rank=np.array(range(self.lambdaa-1,-1,-1))
				
		# Probability Density Function
		l=fitnessScorePop/np.sum(fitnessScorePop)

		# Cumulative Density Function
		cdf=np.cumsum(l)
		for i in range(selectedSize):
			# Choose a random value between 0 and 1
			randomProb=np.random.rand()
			index=self.findIndex(cdf,randomProb)
			rankChoice=rank[index]
			#print(rankChoice,len(fitnessScorePop))
			fitnessRankPopChoice=fitnessRankPop[rankChoice]
			candidateChoice=self.population[fitnessRankPopChoice]
			selected.append(candidateChoice)
		return np.array(selected)

	def slice_inversion(self,gene):
		""" Randomly cut a slice and inverse it"""
		inversePos=np.random.randint(1,self.numCity,2)
		sublist=gene[inversePos[0]:inversePos[1]]
		sublist= np.flip(sublist)
		gene[inversePos[0]:inversePos[1]]=sublist
		return gene
	
	def mutation_inverse(self,offspring):
		""" Inverse Mutation """

		for i in range(len(offspring)):
			if np.random.rand()<=self.mutationProb:
				offspring[i]=self.slice_inversion(offspring[i])
		mutatedOffspring=offspring
		return mutatedOffspring
	
	def mutation_center_inverse(self,offspring):
		""" Center Inversion"""

		for i in range(len(offspring)):
			if np.random.rand()<=self.mutationProb:
				#centrePos=self.numCity//2
				centrePos=np.random.randint(1,self.numCity)
				offspring[i][1:centrePos]=np.flip(offspring[i][1:centrePos])
				offspring[i][centrePos:]=np.flip(offspring[i][centrePos:])
		mutatedOffspring=offspring
		return mutatedOffspring
	
	def mutation_shuffle(self,offspring):
		for i in range(len(offspring)):
			if np.random.rand()<=self.mutationProb:
				inversePos=np.random.randint(1,self.numCity,2)
				offspring[i][inversePos[0]:inversePos[1]]=np.random.permutation(offspring[i][inversePos[0]:inversePos[1]])
		mutatedOffspring=offspring
		return mutatedOffspring
	
	def elimination(self,oldPopulation,offspring):
		"""New Population = Elitism+Offspring"""
		newPopulation=np.array([])
		fitnessScorePop=self.fitness(oldPopulation)
		sortScorePop=np.argsort(fitnessScorePop)
		newPopulation=oldPopulation[sortScorePop[:self.lambdaa//2]]
		newPopulation=np.concatenate((newPopulation,offspring),axis=0)
		
		fitnessScorePop=self.fitness(newPopulation)
		self.bestSolutions.append(np.min(fitnessScorePop))
		self.solutions.append(newPopulation[np.argmin(fitnessScorePop)])
		self.meanSolutions.append(np.mean(fitnessScorePop))
		self.population=newPopulation


	def greedySol(self):
		greedyGene=[np.random.randint(0,self.numCity)]
		#score=0
		while len(greedyGene)<self.numCity-1:
			index=greedyGene[-1]
			added=False
			i=1
			minCity=np.argsort(self.distanceMat[index])
			while not added:
				if minCity[i] not in greedyGene:
					greedyGene.append(minCity[i])
					#score+=self.distanceMat[index][minCity[i]]
					added=True
				i+=1
		extraCity=np.setdiff1d(list(range(self.numCity)),greedyGene)[0]
		greedyGene.append(extraCity)
		while greedyGene[0]!=0:
			greedyGene=np.roll(greedyGene,-1)
		#print(len(np.unique(greedyGene)))
		return greedyGene

	def k_opt(self,gene,k=1,setSize=5):
		""" k-opt Local Search"""
		population=[gene]
		for i in range(setSize):
			population.append(self.slice_inversion(gene))
		fitness=self.fitness(population)
		choice=np.argmin(fitness)
		if choice!=0:
			print("FOUND")
		return population[choice]

	def initialize_population(self,numPopulation=100):
		"""Initialize The Poplulation"""
		population=[]
		greedySize=int(0.2*numPopulation)
		#SampleLocalSize=int(0.2*numPopulation)
		for i in range(greedySize):
			population.append(self.greedySol())

		
		"""for i in range(SampleLocalSize):
			geneNum=np.random.randint(0,greedySize)
			gene=population[geneNum]
			population.append(self.slice_inversion(gene))"""
		for i in range(numPopulation-greedySize):
			population.append(self.permutation())
		
		population=np.array(population)
		population.resize(numPopulation,self.numCity)
		return population

	def permutation(self):
		""" Returns a permuation of C Integers representing a list of cities."""
		gene=np.array([0])
		gene=np.append(gene,np.random.permutation(list(range(1,self.numCity))))
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
	fn.optimize("data/tour750.csv")
	#plt.plot(range(fn.generation),fn.bestSolutions[1:])
	print(fn.solutions[-1])
	plt.show()