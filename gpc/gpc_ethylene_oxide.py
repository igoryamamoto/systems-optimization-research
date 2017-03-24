# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:34:03 2017

@author: Ígor Yamamoto
"""
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# prediction horizon
p = 15
# control horizon
m = 3
# number of inputs
nu = 2
# number of outputs
ny = 2


# tf = -0.19/s
# tf = -1.7/(19.5*s+1) 
# tf = -0.763/(31.8*s+1)
# tf = 0.235/s
