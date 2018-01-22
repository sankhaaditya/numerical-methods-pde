
import sys
import math
import time
import numpy as np
from numpy import linalg
from numpy.linalg import inv
import GridUtils as gu
import DiscUtils as du
import IterUtils as iu
import Multigrid as mg
import scipy


def sourceSetup(S, N):

    # Source blocks defined in 7x7
    # Fineness of our grid then is:

    fine = int((N-1) / 6)

    # Define NxN discretized f vector

    f = np.zeros([N**2])

    # Looping through the 4 blocks
    
    for s in S:

        # Only valid block numbers

        if s >= 1 and s <= 16:

            # What is k of block root point in NxN grid?

            r_k = N * fine * (math.floor((s - 1) / 4) + 1) + fine * ((s - 1) % 4 + 1)
            
            # Looping over grid point falling within block :

            for i in range(0, fine+1):
                for j in range(0, fine+1):

                    # What is k of grid point?
                    
                    p_k = r_k + N * i + j

                    f[p_k] = 1.0

        else:

            print('Source blocks numbers can only be 1 - 16!')
            sys.exit()

    return f


# Grid size (N-1 Divisions, N points) :
# Choices : N = 7, 13, 25

N = 25

if N != 7 and N != 13 and N != 25:
    
    print('N must be 7, 13, or 25 only!')
    sys.exit()

# Get sources at grid points using source blocks
# [1-16]

S = [1, 7, 14, 16]

f = sourceSetup(S, N)

# Apply BC to u and getting mapping to unknown quantities

# eqn, neq = gu.applyBC(u, N)

# Extracting the vector of f at non-boundary points

#b = gu.getNonBound(f, eqn, neq)

# Get A matrix in coordinate form

#A = du.matLaplace(eqn, neq, N)

# Simple Gauss Elimination to find unknowns x

# x = scipy.sparse.linalg.inv(A).dot(b)

# Updating u with x values

#u = gu.updateNonBound(u, x, eqn, neq)

# Perform Jacobi matrix splitting

#RJ, fJ = iu.Jacobi(A, b)

# Perform Gauss-Seidel matrix splitting

#RGS, fGS = iu.GaussSeidel(A, b)

#x = np.zeros([neq])

def m(N, f):
    
    u = np.zeros([N**2])

    u, eqn, neq = gu.applyBC(u, N)

    b = gu.getNonBound(f, eqn, neq)

    A = du.matLaplace(eqn, neq, N)

    x = np.zeros([neq])

    RGS, fGS = iu.GaussSeidel(A, b)

    for t in range(0, 00):
        
            x = RGS.dot(x) + fGS

    u = gu.updateNonBound(u, x, eqn, neq)

    for t in range(0, 5):

        #t0 = time.time()

        u = mg.VCycle(N, u, f, 3)
    
        #t1 = time.time()
        #print(t1-t0)
        
    print(u[26]*24)

def g(N, f):

    u = np.zeros([N**2])

    u, eqn, neq = gu.applyBC(u, N)

    b = gu.getNonBound(f, eqn, neq)

    A = du.matLaplace(eqn, neq, N)

    x = np.zeros([neq])

    RGS, fGS = iu.GaussSeidel(A, b)

    for t in range(0, 350):
        
            x = RGS.dot(x) + fGS

    u = gu.updateNonBound(u, x, eqn, neq)

    print(u[26]*24)

t0 = time.time()
for t in range(0, 1):
    g(N, f)
t1 = time.time()
print(t1-t0)
t0 = time.time()
for t in range(0, 1):
    m(N, f)
t1 = time.time()
print(t1-t0)
    
