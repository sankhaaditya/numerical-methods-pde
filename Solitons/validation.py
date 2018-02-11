import numpy as np
import math as ma
import matplotlib.pyplot as plt
import solitons as sol


def exact(v, t):
    i = 0
    for x in np.arange(-8.0, 8.0, dx):
        u[i] = -v / (2*ma.cosh(ma.sqrt(v)*(x-t*v)/2)**2)
        i += 1

def init2():
    i = 0
    for x in np.arange(-8.0, 8.0, dx):
        v=16
        u[i] = -8*np.exp(-x**2)
        i += 1
        

N = 160
u = np.zeros([N])
dx = 0.1
dt = 0.00001

v = 16
exact(v, 0.0)

T = 0000

for t in range(0, T):
    #print(t)
    u = sol.rk4(u, dx, dt)
    
plt.plot(np.arange(-8.0, 8.0, dx), u)
exact(v, T*dt)
plt.plot(np.arange(-8.0, 8.0, dx), u)
plt.show()
