import numpy as np
from numpy import linalg as la
from numpy.linalg import inv
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
import math
import sys


def applyBC(u, N):

    # To apply a dirichlet BC of 0 to u on the 4 side
    # and create a mapping between a vector of unknowns and u :

    neq = 0
    eqn = []

    #u = np.zeros([N**2])

    # Looping through all grid points :

    for k in range(0, N**2):

        # What are the (i, j) coordinates?

        i = math.floor(k / N)
        j = k % N

        # If it's on a side :

        if i == 0 or j == 0 or i == N-1 or j == N-1:

            # Value is constant and 0
            
            u[k] = 0.0

            # No unknown associated

            eqn.append(-1)

        # Else it's an interior point :

        else:

            # There is an unknown associated

            eqn.append(neq)
            neq += 1

    return u, eqn, neq


def getNonBound(u, eqn, neq):

    # For extracting non-boundary values from u

    # If length of u and eqn are equal :

    if len(u) == len(eqn):

        x = np.zeros([neq])

        for k in range(0, len(u)):

            # Boundary values are assigned -1

            if eqn[k] != -1:

                x[eqn[k]] = u[k]

    else:

        print('u and eqn not of same length!')
        sys.exit()

    return x


def updateNonBound(u, x, eqn, neq):

    # Update u with variable values x

    # If vector lengths are correct :

    if len(u) == len(eqn) and len(x) == neq:

        for k in range(0, len(u)):

            # If it is not a boundary element

            if eqn[k] != -1:

                u[k] = x[eqn[k]]

    else:

        print('Check lengths of vectors!')
        sys.exit()

    return u


def restrict(u, N):

    # Fineness = 1/2

    # u = Array of quantities of interest in N*N grid (1)
    # u2 = np.array([((N-1)/2+1)**2])

    # Need to convert to grid (2)

    # Traversing in (2) :

    N2 = int((N-1)/2+1)

    u2 = np.zeros([N2**2])

    for k2 in range(0, N2**2):

        i2 = math.floor(k2 / N2)
        j2 = k2 % N2

        i = 2 * i2
        j = 2 * j2

        k = N * i + j

        u2[k2] = u[k]

    return u2


def prolongate(u, N):
    
    # Fineness = 2

    # u = Array of quantities of interest in N*N grid (1)
    # u2 = np.array([(2*(N-1)+1)**2])

    # Need to convert to grid (2)

    # Traversing in (1) :

    N2 = (2*(N-1)+1)

    u2 = np.zeros([N2**2])

    for c_k in range(0, N**2):
        
        # Contributions to the grid (2) points near around :

        # 4 Corners (eg. (i+1, j-1)) ((i, j) are in grid (1))

        # 4 Middles of Sides (eg. (i+1, j))

        # and Center (i, j) :

        c_i = math.floor(c_k / N)
        c_j = c_k % N
        
        # k in grid (2)
        c_k2 = 2 * (N2 * c_i + c_j)
        
        # Looping through 9 points around center for contribution :
        
        for p_i in range(c_i-1, c_i+2):
            for p_j in range(c_j-1, c_j+2):
                
                # Does the point exist? :
                
                if p_i >= 0 and p_j >= 0 and p_i < N and p_j < N:
                    
                    # What is k of the point in grid (2)? :

                    p_k2 = c_k2 + N2 * (p_i - c_i) + (p_j - c_j)
                    
                    # If Corners :

                    if p_i != c_i and p_j != c_j:
                        
                        u2[p_k2] += u[c_k] / 4

                    # If Middles of Sides :

                    if (p_i == c_i and p_j != c_j) or (p_i != c_i and p_j == c_j):
                        
                        u2[p_k2] += u[c_k] / 2

                    # If Center :

                    if p_i == c_i and p_j == c_j:
                        
                        u2[p_k2] += u[c_k]

    return u2
