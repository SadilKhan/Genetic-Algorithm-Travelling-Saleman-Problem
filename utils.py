import random
import numpy as np
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
def PMX(parent1,parent2):
    """Partially Mapped Crossover"""
    dictC1C2=dict()
    dictC2C1=dict()
    n=len(parent1)
        
    random_cut=sorted(random.sample(range(1,n),2))
    #print(random_cut)
    #random_cut.append(random_cut[0]+20)
    #print(random_cut)

    child1=[-1]*n
    child2=[-1]*n
    problem=False

    child1[random_cut[0]:random_cut[1]]=parent2[random_cut[0]:random_cut[1]]
    child2[random_cut[0]:random_cut[1]]=parent1[random_cut[0]:random_cut[1]]

    for i in range(random_cut[0],random_cut[1]):
        dictC1C2[child1[i]]=child2[i]
        dictC2C1[child2[i]]=child1[i]
    y=[j for j in range(n) if j not in range(random_cut[0],random_cut[1])]
    for i in y:
        if parent1[i] not in child1:
            child1[i]=parent1[i]
        else:
            try:
                child1[i]=findValue(parent1[i],dictC1C2,dictC2C1,child1,child2)
            except:
                problem=True
                break

        if parent2[i] not in child2:
            child2[i]=parent2[i]
        else:
            try:
                child2[i]=findValue(parent2[i],dictC2C1,dictC1C2,child2,child1)
            except:
                problem=True
                break

    if problem:
        return parent1,parent2
    return child1,child2


def findValue(val,dict12,dict21,c1,c2):
    found=False
    while (not found):
        if dict12[val] not in c1:
            found=True
            return dict12[val]
        else:
            return findValue(dict12[val],dict12,dict21,c1,c2)