import numpy as np
import sys
import time
import math


# Setting up the sources

def sourceSetup(N, S):

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


# Calculate absolute position based on (i, j) in variables domain

def getPos(i, j, N):

    return (N * (i + 1) + j + 1)


def neighPoints(u, i, j, N):

    val = 0.0

    # Point on Left

    if i > 0:

        k_p = getPos(i-1, j, N)

        val += u[k_p]

    # Point on Right

    if i < (N-3):

        k_p = getPos(i+1, j, N)

        val += u[k_p]

    # Point Below

    if j > 0:

        k_p = getPos(i, j-1, N)

        val += u[k_p]

    # Point Above

    if j < (N-3):

        k_p = getPos(i, j+1, N)

        val += u[k_p]

    return val


# Gauss-Seidel cycle

def GaussSeidel(N, u, f):    
    
    # Iterating over internal points (variables)

    for i in range(0, (N-2)):
        for j in range(0, (N-2)):

            # Position
            
            k = getPos(i, j, N)

            val = neighPoints(u, i, j, N)

            # Add force term

            val += (f[k] / (N-1)**2)

            u[k] = val / 4

    return u
