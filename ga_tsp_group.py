import numpy as np
import random
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class TSP:
    """ Genetic Algorithm for Travelling Salesman Problem"""
    def __init__(self,cityList=None,distanceMat=None,generation=400,recombProb=1,mutationProb=0.01):
        self.population = cityList
        self.distanceMat = distanceMat
        self.generation=generation # Number of Generation to proceed
        self.recombProb = recombProb
        self.mutationProb = mutationProb

        self.optimize()

    def optimize(self):
        self.lambdaa = len(self.population) # population Size
        self.meanSolutions=[] # mean of the fitness score per generations
        self.bestSolutions=[] # Value of the best solutions per generations
        self.solutions=[]
        i=0
        
        while i<self.generation:
            i+=1
            selected=self.selection()
            offspring=self.crossover(selected)     
            mutatedOffspring=self.mutation(offspring)

            newPopulation=np.concatenate((self.population,mutatedOffspring),axis=0)

            self.elimination(newPopulation)
        
        print(f"The best solution is {self.bestSolutions[-1]} and route is {self.solutions[-1]}" )

    def fitness(self,population):
        """Calculate the total distance for the route"""

        scorePop=[]
        for gene in population:
            score=0
            for i in range(len(gene)):
                score+=self.distanceMat[gene[i],gene[(i+1)%len(gene)]]
            scorePop.append(score)
        return scorePop

    def mutation(self,offspring):

        """ Swap Mutation"""

        for i in range(len(offspring)):
            if np.random.rand()<=self.mutationProb:
                swapPos=np.random.randint(0,9,2)
                val1,val2=offspring[i][swapPos[0]],offspring[i][swapPos[1]]
                offspring[i][swapPos[0]]=val2
                offspring[i][swapPos[1]]=val1   
        
        mutatedOffspring=offspring
        
        return mutatedOffspring

        
    def crossover(self,selected):
        """ Ordered crossover"""
        offspring=[]
        for i in range((len(selected)//2)-1):
            child=[]
            childP1=[]
            childP2=[]
            random_cut=np.sort(np.random.choice(range(1,9),2))
            parent1=selected[2*i]
            parent2=selected[2*i+1]
            childP1+=list(parent1[random_cut[0]:random_cut[1]])
            childP2=[chr for chr in parent2 if chr not in childP1]
            child=childP2[:random_cut[0]]+childP1+childP2[random_cut[0]:]
            offspring.append(child)
       
        return np.array(offspring)
        
    def selection(self):
        """K tournament Selection"""
        selectedSize=int(self.lambdaa//2)

        selected=[]
        for i in range(selectedSize):
            # Choose random candidates
            choices=self.population[np.random.choice(range(self.lambdaa),3)]

            # Evalute fitness for all candidates
            scores=self.fitness(choices)

            # Choose the best 2 of them
            bestCandidate=choices[np.argsort(scores)[0]]
            selected.append(bestCandidate)
        return np.array(selected)
        
    def elimination(self,newPopulation):
        """ Eliminate 2 least scoring candidates"""
        fitnessScorePop=self.fitness(newPopulation)
        sortScorePop=np.argsort(fitnessScorePop)
        self.bestSolutions.append(np.min(fitnessScorePop))
        self.solutions.append(newPopulation[np.argmin(fitnessScorePop)])
        self.meanSolutions.append(np.mean(fitnessScorePop))
        self.population=newPopulation[sortScorePop[:100]]

        
        

def initialize_population(numPopulation=100):
    population=[]

    for i in range(numPopulation):
        population.append(permutation())
    
    population=np.array(population)
    return population

def permutation():
    return np.random.permutation(list(range(0,10)))

        
        
if __name__="__main__":
    b=np.random.randint(0,100,(10,10))
    distanceMat=(b+b.T)/2
    cityList=initialize_population()
    tsp=TSP(cityList,distanceMat)
    
