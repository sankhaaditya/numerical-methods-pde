import numpy as np
import matplotlib.pyplot as plt

def init(rho):

    for i in range(0, int(len(rho)/2)):

        rho[i] = rho_L

    for i in range(int(len(rho)/2), len(rho)):

        rho[i] = rho_R
            
    return rho

def update(rho):

    def u(i):

        return u_max*(1-rho[i]/rho_max)

    def f(i):

        return rho[i]*u(i)

    def ar(i):

        return u_max*(1-(rho[i]+rho[i+1])/rho_max)

    def al(i):

        return u_max*(1-(rho[i-1]+rho[i])/rho_max)

    def dF(i):

        if i == 0:

            Fl = rho_L*u_max*(1-rho_L/rho_max)
            #Fl = 0.0

        else:

            if scheme == 0:

                Fl = (f(i-1)+f(i))/2 - abs(al(i))*(rho[i]-rho[i-1])/2

            elif scheme == 1:
               
                if rho[i-1] < rho[i]:

                    if f(i-1) < f(i):

                        Fl = f(i-1)

                    else:

                        Fl = f(i)

                else:

                    if f(i-1) > f(i):

                        Fl = f(i-1)

                    else:

                        Fl = f(i)

        if i == (len(rho)-1):

            Fr = rho_R*u_max*(1-rho_R/rho_max)
            #Fr = 0.0

        else:

            if scheme == 0:

                Fr = (f(i)+f(i+1))/2 - abs(ar(i))*(rho[i+1]-rho[i])/2

            elif scheme == 1:

                if rho[i] < rho[i+1]:

                    if f(i) < f(i+1):

                        Fr = f(i)

                    else:

                        Fr = f(i+1)

                else:

                    if f(i) > f(i+1):

                        Fr = f(i)

                    else:

                        Fr = f(i+1)

        return Fr - Fl
    
    rho_new = np.zeros([len(rho)])

    for i in range(0, len(rho)):

        rho_new[i] = rho[i] - dt*(dF(i))/dx

    return rho_new

def update2(rho):

    def u(i):

        return u_max*(1-rho[i]/rho_max)

    def f(i):

        return rho[i]*u(i)

    def char(i):

        return u_max*(1-2*rho[i]/rho_max)

    def F(l, r):
        
        if char(l) < char(r):     # Expansion fan
            
            if char(l) >= 0.0:

                return f(l)

            elif char(r) <= 0.0:

                return f(r)

            else:
                
                return (u_max*rho_max/2)*u_max*(1-u_max/2)

        elif char(l) > char(r):     # Shock wave

            if (f(r)-f(l))/(rho[r]-rho[l]) < 0.0:

                return f(r)

            else:

                return f(l)

        else:
            
            return f(l)

    def dF(i):

        if i == 0:

            Fl = rho_L*u_max*(1-rho_L/rho_max)

        else:

            Fl = F(i-1, i)

        if i == (len(rho)-1):

            Fr = rho_R*u_max*(1-rho_R/rho_max)

        else:
            
            Fr = F(i, i+1)

        return Fr - Fl
    
    rho_new = np.zeros([len(rho)])
    
    for i in range(0, len(rho)):

        rho_new[i] = rho[i] - dt*(dF(i))/dx
        
    return rho_new


rho_max = 1.0
u_max = 1.0
rho_L = 0.8
rho_R = 0.0

L = 4.0
N = 400
dx = L/N
dt = 0.8*dx/u_max

rho = np.zeros([N])

rho = init(rho)

x = np.arange(-2.0+dx/2, 2.0+dx/2, dx)

scheme = 1

t = 0.0

for i in range(0, 200):
    
    rho = update2(rho)

    t += dt

plt.plot(x, rho)
plt.show()



    
