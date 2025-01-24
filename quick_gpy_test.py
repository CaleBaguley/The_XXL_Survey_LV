import time
import pandas as pd
import math as maths
from matplotlib import pyplot as plt
import numpy as np
import copy as cp
import sys
sys.path.insert(0, "/homeb/jb14389")
import GPy

def GenData(count):

    x = [[np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)] for  i in range(count)]
    
    y = [[0] for i in range(count)]
    
    for i in range(count):
    
        if (x[i][0]**2 + 4*x[i][1]**2 < 0.5):
            y[i] = [1]

    return np.asarray(x), np.asarray(y)

x, y = GenData(100)

k = GPy.kern.RBF(3)

m = GPy.models.GPClassification(x, y, kernel = k)
m.optimize()

ard_params = m.kern.input_sensitivity(summarize=False)

print("\n --- ARD paramiter values ------------ \n")
print(ard_params)
print("\n ------------------------------------- \n")

k = GPy.kern.RBF(3, ARD = 1)

m = GPy.models.GPClassification(x, y, kernel = k)
m.optimize()

ard_params = m.kern.input_sensitivity(summarize=False)

print("\n --- ARD paramiter values ------------ \n")
print(ard_params)
print("\n ------------------------------------- \n")

m.kern.plot_ARD()

plt.show()



m.plot_f(visible_dims = [0,1])
plt.scatter(x[:,0],x[:,1], c = y[:,0])
plt.show()

