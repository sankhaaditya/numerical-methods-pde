import numpy as np
from numpy import linalg as la
from numpy.linalg import inv
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
import math
import time


def sourceSetup(N, source, p):

    f = np.zeros([(N-2)**2])

    fineness = int((N-1) / 6)
    
    for k in range(0, len(source)):

        i_orig = int((source[k]-1) / 4) + 1
        j_orig = (source[k]-1) % 4 + 1

        i = i_orig * fineness
        j = j_orig * fineness
 
        for m in range(0, fineness+1):
            for n in range(0, fineness+1):
                
                f[int(p[i+m][j+n])] = 1.0

    return f


def generateMatrix(L, N, source):

    h = L / (N-1)

    p = np.zeros([N, N])
    neq = 0

    for i in range(0, N):
        for j in range(0, N):

            if i == 0 or i == N-1 or j == 0 or j == N-1:
                p[i][j] = -1
            else:
                p[i][j] = neq
                neq += 1

    row = []
    col = []
    data = []

    def append(pc, i, j, d):
        
        if p[i][j] != -1:
            
            row.append(pc)
            col.append(p[i][j])
            data.append(d)
    
    for i in range(1, N-1):
        for j in range(1, N-1):

            pc = p[i][j]
            append(pc, i, j, 4 / h**2)
            append(pc, i-1, j, -1 / h**2)
            append(pc, i+1, j, -1 / h**2)
            append(pc, i, j-1, -1 / h**2)
            append(pc, i, j+1, -1 / h**2)

    A = sparse.coo_matrix((data, (row, col)))
    D = np.diag(np.diag(A.todense()))
    L = np.tril(A.todense()) - D
    U = np.triu(A.todense()) - D

    f = sourceSetup(N, source, p)
    
    return D, L, U, f


def calcRF(R):

    om_opt = 1.0
    minSR = 1.0

    eigR = la.eigvals(R)
    
    for i in range(0, 41):
        om = i * 0.05
        SR = max(abs(om * eigR - om + 1))
        print(om, SR)
        if SR < minSR:
            minSR = SR
            om_opt = om
    
    return 1.0, max(abs(la.eigvals(R)))
    #return om_opt, minSR
        


def iterate(R, b, N):

    om, specRad = calcRF(R)

    u = np.zeros((N-2)**2)
    
    if specRad > 1:
        print('Not Convergent!')
        return

    else:
        r = -1 / math.log(specRad, 10)
        print(r)
        for t in range(0, 5*math.ceil(r)):
            u = (om * R - om + 1).dot(u) + om * b
        
    return u


def jacobi(L, N, source):
    
    D, L, U, f = generateMatrix(L, N, source)

    Dinv = inv(D)
    b = Dinv.dot(f)
    RJ = -Dinv.dot(L+U)

    u = iterate(RJ, b, N)
    
    return u

    
def gs(L, N, source):

    D, L, U, f = generateMatrix(L, N, source)

    DLinv = inv(D+L)
    b = DLinv.dot(f)
    RGS = -DLinv.dot(U)

    u = iterate(RGS, b, N)

    return u
    

L = 1
N = 25
source = [1, 7, 14, 16]
u = gs(L, N, source)
#u = jacobi(L, N, source)
print(u[528] * 24)
