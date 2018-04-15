import numpy as np
import math as ma
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import solitons as sol

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-8, 8), ylim=(-9, 1))
ax.grid()


line1, = ax.plot([], [], '-', label='RK4', lw=2)

ax.legend()
velocity_template = 'V1 = 16, V2 = 4, t = %f'
text = ax.text(0.05, 0.95, '', transform=ax.transAxes)


def init():

    global u
    
    u = np.zeros([N])

    u = pulse1(u, 16, 4.0, 0.0)
    u = pulse1(u, 4, -4.0, 0.0)

    #u = pulse2(u, 0.0)
    
    line1.set_data(np.arange(-8, 8, 0.1), u)
    
    text.set_text('')
    
    return line1, text


def animate(i):

    global u

    u = sol.rk4(u, dx, dt)

    line1.set_data(np.arange(-8, 8, 0.1), u)
    
    text.set_text(velocity_template % (i/1000))
    
    return line1, text


def pulse1(u, v, x0, t):
    
    for x in np.arange(-8.0, 8.0, dx):
        
        u[int((x+8)*10)] += -v / (2*ma.cosh(ma.sqrt(v)*(x-v*t-x0)/2)**2)
        
    return u


def pulse2(u, x0):
    
    for x in np.arange(-8.0, 8.0, dx):
        
        u[int((x+8)*10)] += -8*np.exp(-(x-x0)**2)
        
    return u


def pulse3(u, x0):
    
    for x in np.arange(-8.0, 8.0, dx):
        
        u[int((x+8)*10)] += -6/ma.cosh(x)**2
        
    return u


def pulse4(u, x0):
    
    for x in np.arange(-8.0, 8.0, dx):
        
        if x > -1.0 and x < 1.0:

            u[int((x+8)*10)] += -8*abs(x)-8
        
    return u

N = 160
dx = 0.1
dt = 0.001
T = 2000

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=T, interval=0, blit=True)
plt.show()
