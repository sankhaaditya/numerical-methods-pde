
import math
import sys
import time
import numpy as np
from numpy import linalg as la
from numpy.linalg import inv
import scipy
from scipy import sparse
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix


def getDLU(A):

    # Get the diagonal, lower, upper matrices from A
    
    d = A[0,0]
    D = csr_matrix(csr_matrix.get_shape(A))
    
    for i in range(0, csr_matrix.get_shape(A)[0]):
        D[i, i] = d
    L = scipy.sparse.tril(A) - D
    U = scipy.sparse.triu(A) - D

    return D, L, U


def checkStable(R):
    return True
    # Spectral Radius must be less than 1.0 :

    SR = max(abs(la.eigvals((R.todense()))))

    if SR < 1.0:

        return True

    else:

        return False


def Jacobi(A, f):

    # Find Jacobi matrix and force vector

    # Only for <class 'scipy.sparse.coo.coo_matrix'> A

    if isinstance(A, scipy.sparse.csr_matrix):

        # Get the diagonal, lower, upper matrices from A
        
        D, L, U = getDLU(A)

        # Calculate RJ and fJ
        
        Dinv = np.zeros([len(D), len(D)])
        for i in range(0, len(D)):
            Dinv[i][i] = 1 / D[i][i]
         
        fJ = Dinv.dot(f)
        RJ = -Dinv.dot(L+U)
        
        # Check if RJ is stable

        if checkStable(RJ) == True:

            return RJ, fJ

        else:

            print('RJ not stable!')
            sys.exit()
            
    else:

        print('Matrix A must be scipy.sparse.coo.coo_matrix!')
        sys.exit()


def GaussSeidel(A, f):

    # Find Gauss-Seidel matrix and force vector

    # Only for <class 'scipy.sparse.coo.coo_matrix'> A

    if isinstance(A, scipy.sparse.csr_matrix):

        # Get the diagonal, lower, upper matrices from A
        
        D, L, U = getDLU(A)

        # Calculate RGS and fGS

        DL = D + L

        # Inverting D + L
        
        DLInv = scipy.sparse.linalg.inv(DL)
        fGS = DLInv.dot(f)
        RGS = -DLInv.dot(U)

        # Check if RGS is stable

        if checkStable(RGS) == True:

            return RGS, fGS

        else:

            print('RGS not stable!')
            sys.exit()

    else:

        print('Matrix A must be scipy.sparse.coo.coo_matrix!')
        sys.exit()



