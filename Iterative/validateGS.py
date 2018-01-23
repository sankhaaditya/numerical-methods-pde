import numpy as np
import csv
import math
import solver as s
import matplotlib.pyplot as plt

N = 25

u = np.zeros([N**2])

S = [1, 7, 14, 16]

f = s.sourceSetup(N, S)
 
ifile = open('test.csv')
reader = csv.reader(ifile)

Linf_arr = []
L2_arr = []

for test in range(0, 3):
    
    for t in range(0, 100*(test+1)):
        u = s.GaussSeidel(N, u, f)

    res = np.zeros([N, 4])
    for i in range(0, N):
        res[i, 0] = u[N+i]*24
        res[i, 1] = u[N*(N-2)+i]*24
        res[i, 2] = u[N*i+1]*24
        res[i, 3] = u[N*i+N-2]*24

    r = 0
    Linf = 0.0
    L2 = 0.0
    for row in reader:
        for c in range(0, 4):
            e = abs(float(row[c]) - res[r][c])
            if e > Linf:
                Linf = e
            L2 += e**2
        r += 1
    L2 = math.sqrt(L2)

    Linf_arr.append(Linf)
    L2_arr.append(L2)
 
ifile.close()

f, axarr = plt.subplots(2, sharex=True)
plt.title('Gauss_Seidel, N=25')
axarr[0].plot(range(100, 400, 100), Linf_arr, '-o')
axarr[0].set_title('Linf')
axarr[1].plot(range(100, 400, 100), L2_arr, '-o')
axarr[1].set_title('L2')
plt.xlabel('No. of Iterations')
plt.show()
