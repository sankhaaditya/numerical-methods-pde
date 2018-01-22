
import math
import sys
import numpy as np
import scipy as sp
from scipy import sparse


def matLaplace(eqn, neq, N):

    # Create a stiffness matrix for Laplace operator
    # using central difference

    # Number of unknowns : neq

    # Sparse Matrix :

    row = []
    col = []
    data = []

    def addtoMat(p_eq):
        if p_eq != -1:
            row.append(c_eq)
            col.append(p_eq)
            data.append(-1 / h**2)

    # Grid Width :

    h = 1 / (N-1)

    # Looping over the all grid points :

    for c_k in range(0, N**2):

        # Is point c an unknown?
        
        c_eq = eqn[c_k]
        
        if c_eq != -1:

            # (c, c) will always exist

            row.append(c_eq)
            col.append(c_eq)
            data.append(4 / h**2)

            # What is (i, j) at c?

            i = math.floor(c_k / N)
            j = c_k % N

            # If point p below exists :

            if j > 0:

                # If p is also an unknown :

                p_eq = eqn[c_k-1]

                addtoMat(p_eq)

            # If point p above exists :

            if j < N-1:

                # If p is also an unknown :

                p_eq = eqn[c_k+1]

                addtoMat(p_eq)

            # If point p at left exists :

            if i > 0:

                # If p is also an unknown :

                p_eq = eqn[c_k-N]

                addtoMat(p_eq)

            # If point p at right exists :

            if i < N-1:

                # If p is also an unknown :

                p_eq = eqn[c_k+N]

                addtoMat(p_eq)

    A = sparse.coo_matrix((data, (row, col)))

    A = A.tocsr()

    return A
