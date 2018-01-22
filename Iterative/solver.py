import numpy as np
from numpy import linalg as la
from numpy.linalg import inv
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
import itertools as it

# inputs
fineness = 4
#source = [1, 7, 14, 16]
om = 1.4

N = (fineness * 6) + 1
h = 1 / (N-1)
dt = h**2 / 4
u = np.zeros([N, N])
f = np.zeros([N, N])


def sourceSetup(source):
    
    global f
    f = np.zeros([N, N])
    
    for k in range(0, len(source)):

        i_orig = int((source[k]-1) / 4) + 1
        j_orig = (source[k]-1) % 4 + 1

        i = i_orig * fineness
        j = j_orig * fineness
 
        for m in range(0, fineness+1):
            for n in range(0, fineness+1):
                
                f[i+m][j+n] = 1.0
                

def update():

    for i in range(1, N-1):
        for j in range(1, N-1):
    
            u[i][j] = om * (u[i][j] + dt * ( u[i-1][j-1] + u[i+1][j-1] + u[i-1][j+1] + u[i+1][j+1] - 4 * u[i][j] ) / h**2 + dt * f[i][j]) + (1 - om) * u[i][j]

                

testList = list(it.combinations(range(1, 17), 4))
for i in range(0, len(testList)):

    print(i)
    
    sourceSetup(testList[i])
    
    for t in range(0, 3000):
        update()
        if error < error_old:
            error_old = error
        else:
            break

    if error_old < 1.0:
        print(testList[i], t, error_old)
    
#for i in range(0, N):
#    print(i+1, 2 * u[1][i] / h, 2 * u[N-2][i] / h, 2 * u[i][1] / h, 2 * u[i][N-2] / h)
