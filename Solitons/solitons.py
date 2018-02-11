import numpy as np
import math as ma
import matplotlib.pyplot as plt


def f(u, dx):
    
    N = len(u)

    x = np.zeros([N])

    for i in range(0, N):

        x[i] += (u[(i-2)%N]-2*u[(i-1)%N]+2*u[(i+1)%N]-u[(i+2)%N])/(2*dx**3)

        x[i] += 6*u[i]*(u[(i+1)%N]-u[(i-1)%N])/(2*dx)

    return x


def a(u, dx, dt):

    dtfu = dt*f(u, dx)

    a1 = dtfu

    a2 = dtfu + dt*f(a1/2, dx)
    
    a3 = dtfu + dt*f(a2/2, dx)

    a4 = dtfu + dt*f(a3, dx)

    return (a1+2*a2+2*a3+a4)/6


def rk4(u, dx, dt):

    u += a(u, dx, dt)

    return u
