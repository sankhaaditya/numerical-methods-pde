import numpy as np
from numpy import linalg as la
from numpy.linalg import inv
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
import math
import time


def genGrid(l, N):
    
    h = l / (N-1)

    p = np.zeros([N, N])
    neq = 0

    for i in range(0, N):
        for j in range(0, N):

            if i == 0 or i == N-1 or j == 0 or j == N-1:
                p[i][j] = -1
            else:
                p[i][j] = neq
                neq += 1

    return p, h


def sourceSetup(N, p, source):

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


def generateMatrix(N, p, h):

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
    
    return A, D, L, U


def calcRF(R):

    if optOm == True:
        
        om_opt = 1.0
        minSR = 1.0
        
        for i in range(0, 11):
            om = 1.0 + i * 0.005
            SR = max(abs(la.eigvals((om * R - om + 1))))
            print(om, SR)
            if SR < minSR:
                minSR = SR
                om_opt = om
        
        return om_opt, minSR

    else:

        return 1.0, max(abs(la.eigvals(R)))


def checkStable(SR):

    if SR >= 1:
        print('Not Convergent!')
        return False
    
    else:
        return True


def restrict(N, x):

    N_coarse = (((N-1)/2+1)-2)

    y = np.zeros(N_coarse**2)

    for i in range(0, N_coarse):
        for j in range(0, N_coarse):
            y[N_coarse*i+j] = x[N*(2*i+1)+2*j+1]

    return y


def propagate(N, x):

    N_coarse = (((N-1)/2+1)-2)

    y = np.zeros((N-2)**2)

    for i in range(0, N_coarse):
        for j in range(0, N_coarse):
            y[N_coarse*i+j] = x[N*(2*i+1)+2*j+1]

    return y

    
def calc(l, N, A, R, f, b, mg_levels):
    
    u = np.zeros((N-2)**2)

    om, SR = calcRF(R)

    if checkStable(SR) == True:

        if mg == False:
            
            print('Without multigrid. Iterations required :')
            
            r = -1 / math.log(SR, 10)
            print(r)
            for t in range(0, 5*math.ceil(r)):
                u = (om * R - om + 1).dot(u) + om * b

        else:
            
            print('Using multigrid')
            
            if mg_levels == 0:
                u = inv(A).dot(f)

            else:
                for t in range(0, 2):
                    u = (om * R - om + 1).dot(u) + om * b
                    te = A.dot(u) - f
                    te = restrict(N, te) #Restrict
                    e = jacobi(l, (N-1)/2+1, te, mg_levels-1)
                    e = propagate(N, e) #Propagate
                    u = u + e
                
    return u


def jacobi(l, N, f, mg_levels):
    
    A, D, L, U = generateMatrix(N, genGrid(l, N))

    Dinv = inv(D)
    b = Dinv.dot(f)
    RJ = -Dinv.dot(L+U)

    u = calc(l, N, A, RJ, f, b, mg_levels)
    
    return u

    
def gs(l, N, f, mg_levels):

    p, h = genGrid(l, N)

    A, D, L, U = generateMatrix(N, p, h)

    DLinv = inv(D+L)
    b = DLinv.dot(f)
    RGS = -DLinv.dot(U)

    u = calc(l, N, A, RGS, f, b, mg_levels)

    return u


def solve(l, N, source, method, mg_levels):

    p, h = genGrid(l, N)

    f = sourceSetup(N, p, source)
    
    if method == 0:
        
        print('Jacobi')
        u = jacobi(l, N, f, mg_levels)
        
    elif method == 1:
        
        print('Gauss-Seidel')
        u = gs(l, N, f, mg_levels)

    return u


l = 1
N = 25
source = [1, 7, 14, 16]

method = 1
optOm = False
mg = False
mg_levels = 0

u = solve(l, N, source, method, mg_levels)

print('Output :')
print(u[528] * 24)
