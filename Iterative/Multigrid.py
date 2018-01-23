import numpy as np
import sys
import time
import math as m
import solver as s


def vCycle(N, u, f, level):

    # Iterate 3 times on (h) :

    for t in range(0, 3):
        u = s.GaussSeidel(N, u, f)

    # If it has reached (8h) :
    
    if level == 0:
        return u

    # Calculate truncation error on (2h)

    N2 = int((N - 1) / 2) + 1

    tr2 = np.zeros([N2**2])

    for i2 in range(0, (N2-2)):
        for j2 in range(0, (N2-2)):

            # (i, j) mapping

            i = 2 * i2 + 1
            j = 2 * j2 + 1

            # Positions

            k2 = s.getPos(i2, j2, N2)
            k = s.getPos(i, j, N)

            # Evaluating matrix equation

            val = s.neighPoints(u, i, j, N)

            val -= 4 * u[k]

            val *= (N-1)**2

            val += f[k]

            tr2[k2] = val

    # Initial error guess

    e2 = np.zeros([N2**2])
            
    e2 = vCycle(N2, e2, tr2, level-1)

    # Prolongate e2 and correct u

    for i in range(0, (N-2)):
        for j in range(0, (N-2)):

            k = s.getPos(i, j, N)
            
            i2 = (i - 1) / 2
            j2 = (j - 1) / 2
            
            if i % 2 == 1 and j % 2 == 1:
                # If i odd, j odd

                u[k] += e2[s.getPos(int(i2), int(j2), N2)]

            if i % 2 == 0 and j % 2 == 1:
                # If i even, j odd

                u[k] += (e2[s.getPos(m.floor(i2), int(j2), N2)]
                        + e2[s.getPos(m.ceil(i2), int(j2), N2)]) / 2

            if i % 2 == 1 and j % 2 == 0:
                # If i odd, j even
                u[k] += (e2[s.getPos(int(i2), m.floor(j2), N2)]
                        + e2[s.getPos(int(i2), m.ceil(j2), N2)]) / 2

            if i % 2 == 0 and j % 2 == 0:
                # If i even, j even

                u[k] += (e2[s.getPos(m.floor(i2), m.floor(j2), N2)]
                        + e2[s.getPos(m.ceil(i2), m.ceil(j2), N2)]
                        + e2[s.getPos(m.ceil(i2), m.floor(j2), N2)]
                        + e2[s.getPos(m.floor(i2), m.ceil(j2), N2)]) / 4

    # Iterate 3 more times on (h) :

    for t in range(0, 3):
        u = s.GaussSeidel(N, u, f)

    return u
