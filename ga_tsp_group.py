import numpy as np
import random
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class ga_tsp:
    """ Genetic Algorithm for Travelling Salesman Problem"""
    def __init__(self,lambdaa=100,mu=100,generation=400):
        self.lambdaa = lambdaa # population Size
        self.mu=mu # Offspring Size
        self.generation=generation # Number of Generation to proceed

    def optimize(self):
        
        # Initialize Population
        self.population()

        for i in range(self.generation):
            self.selection()
            self.crossover()
            self.mutation()
            self.evaluation()
            self.elimination()

    def population(self):
         """ Function for initializing Population """
         self.popList=[]
         for i in range(self.lambdaa):
             individual=np.random.permutation(cityClass)
             self.popList.append(individual)
         return self.popList
        

    def selection(self):
        pass
    def mutation(self):
        pass
    def evaluation(self):
        pass
    def elimination(self):
        pass
    def crossover(self):
        pass
