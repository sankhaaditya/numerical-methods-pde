import numpy as np
import csv
import math
import time
import solver as s
import multigrid as m
import matplotlib.pyplot as plt


def calcL2(u):

    res = np.zeros([N, 4])
    for i in range(0, N):
        res[i, 0] = u[N+i]*24
        res[i, 1] = u[N*(N-2)+i]*24
        res[i, 2] = u[N*i+1]*24
        res[i, 3] = u[N*i+N-2]*24
    
    r = 0
    L2 = 0.0
    for row in reader:
        for c in range(0, 4):
            e = abs(float(row[c]) - res[r][c])
            L2 += e**2
        r += 1
    L2 = math.sqrt(L2)

    return L2


N = 25

u = np.zeros([N**2])

S = [1, 7, 14, 16]

f = s.sourceSetup(N, S)
 
ifile = open('test.csv')
reader = csv.reader(ifile)

t0 = time.time()
for t in range(0, 2):
    u = m.vCycle(N, u, f, 3)
t1 = time.time()
L2_vc = calcL2(u)
print(L2_vc, t1-t0)
ifile.close()
ifile = open('test.csv')
reader = csv.reader(ifile)

u = np.zeros([N**2])
t0 = time.time()
for t in range(0, 245):
    u = s.GaussSeidel(N, u, f)

t1 = time.time()
print(calcL2(u), t1-t0)
