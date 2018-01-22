
import math
import sys
import time
import numpy as np
from numpy import linalg as la
from numpy.linalg import inv
import scipy
from scipy import sparse
from scipy.sparse.linalg import inv
import GridUtils as gu
import DiscUtils as du
import IterUtils as iu


def VCycle(N, u, f, level):

    u, eqn, neq = gu.applyBC(u, N)

    b = gu.getNonBound(f, eqn, neq)
    
    A = du.matLaplace(eqn, neq, N)
    
    x = gu.getNonBound(u, eqn, neq)
    
    # Find RGS and fGS
    
    RGS, fGS = iu.GaussSeidel(A, b)
    
    # Iterate thrice using Gauss-Seidel :

    for t in range(0, 3):
        
        x = RGS.dot(x) + fGS

    if level > 0:

        # Get residual :
        
        tr = A.dot(x) - b
        
        # Residual for all grid points :

        tr_all = np.zeros([N**2])

        tr_all = gu.updateNonBound(tr_all, tr, eqn, neq)

        # Restrict to half fineness
        
        tr_all = gu.restrict(tr_all, N)

        N2 = int((N-1)/2+1)

        # Recurse for tr_all

        e = np.zeros([N2**2])
        
        e = VCycle(N2, e, - tr_all, level - 1)
        
        # Propagate e
        
        e = gu.prolongate(e, N2)
        
        # Correct u with e

        u = u + e

        # Get x from u

        x = gu.getNonBound(u, eqn, neq)

        # Iterate thrice more

        for t in range(0, 3):
        
            x = RGS.dot(x) + fGS

    #else:

        # Direct calculation

        #x = scipy.sparse.linalg.inv(A).dot(b)
        
    u = gu.updateNonBound(u, x, eqn, neq)

    return u
