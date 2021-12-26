import Reporter
import numpy as np
import matplotlib.pyplot as plt
import random


class r0873969:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

		# Best Solution yet in tour750: 117388
		# Best Solution in tour250: 46167

	# The evolutionary algorithm's main loop
    def optimize(self, filename):
		# Read distance matrix from file.		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Initialize necessary variables

        # Creating object of the operators
        selection=Selection()
        crossover=Crossover()
        initialize=Initialization()
        mutation=Mutation()
        elimination=Elimination()

        # There is Inf in the distance matrix, replace them with max value*1000
        self.distanceMat = distanceMatrix
        self.noinfmax=self.distanceMat[self.distanceMat!=np.inf].max()*1000
        self.distanceMat=np.nan_to_num(self.distanceMat,copy=True,posinf=self.noinfmax)
        self.numCity=len(self.distanceMat) # Number of cities in the problem.
        # Number of population size
        if self.numCity>100:
            self.population = initialize.initialize_population(min(600,int(self.numCity*0.8)),self.distanceMat)
        else:
            self.population =initialize.initialize_population(100,self.distanceMat)

        self.generation=1 # Number of Iteration
        self.recombProb = 1
        self.mutationProb = 0.6
        self.localSearchProb=0.01
        i=0

        self.lambdaa = len(self.population) # population Size
        self.meanSolutions=[] # mean of the fitness score per generations
        self.bestSolutions=[] # Value of the best solutions per generations
        self.solutions=[] # The candidate solutions
        timeleft=300 # The total time. 5 mins

        

        # Your code here.
        while(True):
            # Selection Operation
            selected=selection.start(population=self.population,selectedSize=int(3*self.lambdaa//4),distanceMat=self.distanceMat)
            # Recombination Operation
            offspring=crossover.start(parentPool=selected,method="OX",distanceMat=self.distanceMat)   
            # Mutation Operation
            mutationFn=np.random.rand()
            if mutationFn>0.4:
                mutatedOffspring=mutation.start(population=offspring,method="INV",mutationProb=self.mutationProb)
            else:
                mutatedOffspring=mutation.start(population=offspring,method="CINV",mutationProb=self.mutationProb)

            # Elimination Operation
            self.population,bs,ms,s=elimination.start(oldPopulation=self.population,offspring=mutatedOffspring,distanceMat=self.distanceMat,timeleft=timeleft)
            self.bestSolutions.append(bs)
            self.meanSolutions.append(ms)
            self.solutions.append(s)
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
            print(f"Generation: {i}, Best Solution:{self.bestSolutions[-1]}, Mean Solution:{self.meanSolutions[-1]},TimeLeft:{timeLeft}")
            if timeLeft < 0:
                break
            i+=1

        # Your code here.
        return 0    

class Selection():
    """ Custom class for Rank Selection Method
    """

    def __init__(self):
        pass
    def start(self,population,selectedSize,distanceMat):
        """
        Rank Selection with Linear Decay

        Parameters
        ----------

        population: 2d array like. Population with candidate solutions.
        
        selectedSize: int. The size of selected candidates in the population that will be 
                        used for crossover

        distanceMat: 2d array like. Distance Matrix

        Returns
        -------

        selected: 2d array. Contains the candidate solutions for crossover
        
        """
        selected=[]
        lambdaa=len(population)
        # Calculate Fitness Scores
        fitnessScorePop=Metrics().fitness(population,distanceMat)
        # Assign rank to the fitness
        fitnessRankPop=np.argsort(fitnessScorePop)
        rank=np.array(range(lambdaa-1,-1,-1))
                
        # Probability Density Function
        l=fitnessScorePop/np.sum(fitnessScorePop)

        # Cumulative Density Function
        cdf=np.cumsum(l)
        while len(selected)<selectedSize:
            randomProb=np.random.rand() # Random value between 0 and 1.
            index=self.findIndex(cdf,randomProb) # The last index of the array is smaller than the value.
            rankChoice=rank[index] # The rank of that index.
            fitnessRankPopChoice=fitnessRankPop[rankChoice] # The index of the candidate for the rank.
            candidateChoice=population[fitnessRankPopChoice] # The candidate.
            # Check if the candidate is present in the solution or not
            selected.append(candidateChoice) # Add the solution
        return np.array(selected)
    
    def findIndex(self,array,value):
        """
        Returns the last index of the array that is smaller than the value

        Parameters
        ----------

        array: 1d array or list like.

        value: int or float.

        Returns
        -------

        index: int. Last Index of the array that is smaller than the value.
        
        """
        for i in range(len(array)):
            if (array[i]>value) & (i==0):
                return 0
            elif array[i]>value:
                return i-1

class Crossover():
    """ Recombination Operator"""
    def __init__(self):
        self.localSearchProb=0.01 # Local Search Probability
    def start(self,parentPool,method,distanceMat):
        """
        Crossover

        Parameters
        ----------

        parentPool: 2d array like. Parent candidate solutions.
        
        method: string , default = "OX". 
                If "OX"--> Ordered Crossover
                If "ERO" --> Edge Recombination Operator

        Returns
        -------
        offspring: 2d array like.        
        """
        self.numCity=len(distanceMat)
        if method == "OX":
            recomOper=self.ordered_crossover
        offspring=[]
        selectedSize=len(parentPool)
        bestCandCrossoverSize=int(0.2*selectedSize) # No of Best candidates used for intra-group crossover
        bestAvgCandCrossoverSize=(selectedSize-2*bestCandCrossoverSize)//4 # No of the candidates used for inter-group crossover

        # Crossover within best candidates
        for i in range(bestCandCrossoverSize):
            parent1=parentPool[2*i]
            parent2=parentPool[2*i+1]
            child1=recomOper(parent1,parent2,distanceMat)
            child2=recomOper(parent2,parent1,distanceMat)
            offspring.append(child1)
            offspring.append(child2)
        # Crossover between Best candidates and rest of the candidates
        for i in range(bestAvgCandCrossoverSize):
            x1=np.random.choice(range(bestCandCrossoverSize))
            x2=np.random.choice(range(bestCandCrossoverSize,selectedSize))
            parent1=parentPool[x1]
            parent2=parentPool[x2]
            child=recomOper(parent1,parent2,distanceMat)
            offspring.append(child)
        
        # Crossover within poor performing candidates
        for i in range(selectedSize-2*bestCandCrossoverSize-bestAvgCandCrossoverSize):
            x1,x2=np.random.choice(range(bestCandCrossoverSize,selectedSize),2)
            parent1=parentPool[x1]
            parent2=parentPool[x2]
            child=recomOper(parent1,parent2,distanceMat)
            offspring.append(child)
        return offspring



    def ordered_crossover(self,parent1,parent2,distanceMat):
        """ 
        Ordered Crossover

        Parameters
        ----------
        parent1: array like.
        
        parent2: array like.

        distanceMat: 2d array like. Distance matrix. 

        Returns
        -------
        child: array like.
        """
        localsearch=LocalSearch()
        child=[]
        childP1=[]
        childP2=[]
        random_cut=np.sort(np.random.choice(range(1,self.numCity),2))
        childP1+=list(parent1[random_cut[0]:random_cut[1]])
        childP2=[chr for chr in parent2 if chr not in childP1]
        child=childP2[:random_cut[0]]+childP1+childP2[random_cut[0]:]
        if np.random.rand()<=self.localSearchProb:
            child=localsearch.k_opt(child,distanceMat=distanceMat)

        return child



class Mutation():
    def __init__(self):
        """Mutation Operation"""
        pass
    def start(self,population,method,mutationProb):

        """ 
        
        Parameters
        ----------

        population: 2d array like
        
        method: string. Mutation type.
                If "INV" --> Inversion Mutation
                If "CINV" --> Center Inversion Mutation

        Returns
        -------
        mutatedOffspring: 2d array like.
        """
        
        if method=="INV":
            mutOper=self.mutation_inverse
        else:
            mutOper=self.mutation_center_inverse
        
        mutatedOffspring=[]
        for i in range(len(population)):
            if np.random.rand()<=mutationProb:
                mutatedGene=mutOper(population[i])
                mutatedOffspring.append(mutatedGene)
        return np.array(mutatedOffspring)
        
    def mutation_center_inverse(self,gene):
        """
        Center Inversion Mutation. The gene is divided into two segments and each part is inversed.

        Parameters
        ----------
        gene: array like.

        Returns
        -------
        gene: mutated gene
        

        """
        numCity=len(gene)
        centrePos=numCity//2
        centrePos=np.random.randint(1,numCity)
        gene[1:centrePos]=np.flip(gene[1:centrePos])
        gene[centrePos:]=np.flip(gene[centrePos:])

        return gene
    def mutation_inverse(self,gene):
        """ 
        Inverse mutation

        Parameters
        ----------
        gene: array like.

        Returns
        -------
        gene: mutated gene
        
        """
        numCity=len(gene)
        inversePos=np.random.randint(1,numCity,2)
        sublist=gene[inversePos[0]:inversePos[1]]
        sublist= np.flip(sublist)
        gene[inversePos[0]:inversePos[1]]=sublist
        return gene

class Elimination():
    def __init__(self):
        self.initialize=Initialization()
    def start(self,oldPopulation,offspring,distanceMat,timeleft):
        """New Population = Elitism+Offspring"""
        lambdaa=len(oldPopulation)
        newPopulation=np.array([])
        fitnessScorePop=Metrics().fitness(oldPopulation,distanceMat)
        sortScorePop=np.argsort(fitnessScorePop)
        newPopulation=oldPopulation[sortScorePop[:lambdaa//4]]

        # Crowding
        if timeleft<100:
            addedOffspring=[]
            for i in range(len(offspring)):
                numSeq=np.random.randint(1,len(newPopulation))
                numLen=np.random.randint(5,20)
                seq=self.calculate_similarity(newPopulation[numSeq:numSeq+numLen],offspring[i])
                newPopulation[numSeq:numSeq+numLen]=seq
                addedOffspring.append(i)
            offspring=np.delete(offspring,addedOffspring,axis=0)
        newPopulation=np.concatenate((newPopulation,offspring),axis=0)

        if timeleft<100:
            ranPopulation=self.initialize.initialize_population(self.lambdaa-len(newPopulation))
            newPopulation=np.concatenate((newPopulation,ranPopulation),axis=0)
        fitnessScorePop=Metrics().fitness(newPopulation,distanceMat)
        bestSolutions=np.min(fitnessScorePop)
        solutions=newPopulation[np.argmin(fitnessScorePop)]
        meanSolutions=np.mean(fitnessScorePop)
        return newPopulation,bestSolutions,meanSolutions,solutions
    
    def calculate_similarity(self,population,gene):
        """ Check if there is any similar sequence as gene and then replace it"""
        for i,p in enumerate(population):
            sim=self.similarity_function(p,gene)
            if sim>=0.75:
                population[i]=gene
                break
        return population
    
    def similarity_function(self,gene1,gene2):
        """ Calculate the similarity between two sequences"""
        difference=gene1-gene2
        return len(difference[difference==0])/len(difference)
    



class LocalSearch():
    """Local Search Methods"""
    def __init__(self):
        pass
    def k_opt(self,gene,k=1,setSize=5,distanceMat=None):
        """ k-opt Local Search"""
        mutation=Mutation()
        metrics=Metrics()

        population=[gene]
        for i in range(setSize):
            population.append(mutation.mutation_inverse(gene))
        fitness=metrics.fitness(population,distanceMat)
        choice=np.argmin(fitness)
        return population[choice]

class Initialization():

    """Advanced Initialization Methods """
    def __init__(self):
        pass
    def initialize_population(self,popSize,distanceMat):
        """ 
        parameters
        ----------

        popSize: int. Population size

        distanceMat: 2d array-like. Distance Matrix

        Returns
        -------
        population: 2d array like.

        """
        self.distanceMat=distanceMat
        self.numCity=len(self.distanceMat)

        population=[]
        greedySize=int(0.3*popSize) # Size of greedy solutions
        for i in range(greedySize):
            population.append(self.greedySol())
        # Add random solutions
        for i in range(popSize-greedySize):
            population.append(self.permutation())
        
        population=np.array(population)
        population.resize(popSize,self.numCity)
        return population
    
    def permutation(self):
        """ Returns a permuation of C Integers representing a list of cities."""
        gene=np.array([0])
        gene=np.append(gene,np.random.permutation(list(range(1,self.numCity))))
        return gene
    def greedySol(self):
	""" Return a greedy like solution """
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


class Metrics():
	""" Score Metrics """

    def __init__(self):
        pass
    def fitness(self,population,distanceMat):
        """Calculate the total distance for the path"""
        scorePop=[]
        for gene in population:
            try:
                score=distanceMat[gene,np.roll(gene,-1)].sum()
            except:
                print("ERROR IN CALCULATING FITNESS SCORE")
                break
            scorePop.append(score)
        return scorePop



if __name__=="__main__":
    main=r0873969()
    main.optimize("./tour750.csv")
