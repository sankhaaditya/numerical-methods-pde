import numpy as np
from numpy import linalg
from numpy.linalg import inv
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Inputs
l = 3.0
b = 0.5
h = 1.0
n = m = 21   # Grid size


neq = 0
p = np.zeros([n, m, 4])
row = []
col = []
data = []
xyGeo = []
deta = 1 / (n-1)
dxi = 1 / (m-1)
u = np.zeros([n, m])
x = np.zeros([n, m])
y = np.zeros([n, m])

def geoSetup():

    global neq

    d = (l - b) / 2
    a = math.sqrt(d**2 - h**2)
    
    # Points A-D in x-y domain
    xyGeo.append([0, h])
    xyGeo.append([0, 0])
    xyGeo.append([b/2, 0])
    xyGeo.append([a+b/2, h])

    # Defining eta-xi domain
    for i in range(0, n):
        for j in range(0, m):
            p[i][j][0] = -0.5 + i * deta
            p[i][j][1] = -0.5 + j * dxi
            if (i == 0 and j != 0) or (j == (m-1) and i != (n-1)):  # Left and top faces
                p[i][j][2] = 1
                p[i][j][3] = neq
                neq += 1
            elif j == 0 or i == (n-1):                              # Bottom and right walls
                p[i][j][2] = 2
                p[i][j][3] = -1
            else:                                                   # Interior point
                p[i][j][3] = neq
                neq += 1
            

# Calculate derivatives for transformation at eta-xi coordinates
def calcCoef(i, j):
    
    xA = xyGeo[0][0]
    xB = xyGeo[1][0]
    xC = xyGeo[2][0]
    xD = xyGeo[3][0]
    yA = xyGeo[0][1]
    yB = xyGeo[1][1]
    yC = xyGeo[2][1]
    yD = xyGeo[3][1]

    eta = p[i][j][0]
    xi = p[i][j][1]
    
    xe = (- xA - xB + xC + xD) / 2 + (- xA + xB - xC + xD) * xi
    xn = (xA - xB - xC + xD) / 2 + (- xA + xB - xC + xD) * eta
    xee = xnn = 0
    xen = - xA + xB - xC + xD

    ye = (- yA - yB + yC + yD) / 2 + (- yA + yB - yC + yD) * xi
    yn = (yA - yB - yC + yD) / 2 + (- yA + yB - yC + yD) * eta
    yee = ynn = 0
    yen = - yA + yB - yC + yD

    J = xe*yn - xn*ye
    a = xn**2 + yn**2
    b = xe*xn + ye*yn
    c = xe**2 + ye**2
    d = (-2*b*yen*xn + 2*b*xen*yn)/J
    e = (-2*b*xen*ye + 2*b*yen*xe)/J

    return a, b, c, d, e, J


# Building matrix


def append(pc, i, j, d):
    if p[i][j][3] != -1:
        row.append(pc)
        col.append(int(p[i][j][3]))
        data.append(d)


def fillMat():
    for i in range(0, n):
        for j in range(0, m):
            
            if p[i][j][2] == 2:     # Wall
                continue

            pc = int(p[i][j][3])
            
            a, b, c, d, e, J = calcCoef(i, j)

            # For uee and ue
            if i == 0:                                                  # At any left most point
                
                #uee = (u[i][j] - 2u[i+1][j] + u[i+2][j]) / deta**2
                append(pc, i, j, - a / J**2 / deta**2)
                append(pc, i+1, j, 2 * a / J**2 / deta**2)
                append(pc, i+2, j, - a / J**2 / deta**2)
                
                if p[i][j][2] != 1:                                     # If no Neumann, otherwise zero
                    
                    #ue = (u[i+2][j] - u[i][j]) / 2 / deta
                    append(pc, i, j, d / J**2 / 2 / deta)
                    append(pc, i+2, j, - d / J**2 / 2 / deta)
                    
                    
            elif i == n-1:                                              # At any right most point
                
                #uee = (u[i-2][j] - 2u[i-1][j] + u[i][j]) / deta**2
                append(pc, i, j, - a / J**2 / deta**2)
                append(pc, i-1, j, 2 * a / J**2 / deta**2)
                append(pc, i-2, j, - a / J**2 / deta**2)
                
                if p[i][j][2] != 1:
                    
                    #ue = (u[i][j] - u[i-2][j]) / 2 / deta
                    append(pc, i, j, - d / J**2 / 2 / deta)
                    append(pc, i-2, j, d / J**2 / 2 / deta)
                    
            else:
                
                #uee = (u[i-1][j] - 2u[i][j] + u[i+1][j]) / deta**2
                append(pc, i, j, 2 * a / J**2 / deta**2)
                append(pc, i+1, j, - a / J**2 / deta**2)
                append(pc, i-1, j, - a / J**2 / deta**2)
                
                #ue = (u[i+1][j] - u[i-1][j]) / 2 / deta
                append(pc, i+1, j, - d / J**2 / 2 / deta)
                append(pc, i-1, j, d / J**2 / 2 / deta)


            # For unn and un
            if j == 0:                                                  # At any bottom most point
                
                #unn = (u[i][j] - 2u[i][j+1] + u[i][j+2]) / dxi**2
                append(pc, i, j, - c / J**2 / dxi**2)
                append(pc, i, j+1, 2 * c / J**2 / dxi**2)
                append(pc, i, j+2, - c / J**2 / dxi**2)
                
                if p[i][j][2] != 1:

                    #un = (u[i][j+2] - u[i][j]) / 2 / dxi
                    append(pc, i, j, e / J**2 / 2 / dxi)
                    append(pc, i, j+2, - e / J**2 / 2 / dxi)
                    
            elif j == m-1:                                              # At any top most point
                
                #unn = (u[i][j-2] - 2u[i][j-1] + u[i][j]) / dxi**2
                append(pc, i, j, - c / J**2 / dxi**2)
                append(pc, i, j-1, 2 * c / J**2 / dxi**2)
                append(pc, i, j-2, - c / J**2 / dxi**2)
                
                if p[i][j][2] != 1:
                    
                    #un = (u[i][j] - u[i][j-2]) / 2 / dxi
                    append(pc, i, j, - e / J**2 / 2 / dxi)
                    append(pc, i, j-2, e / J**2 / 2 / dxi)
                    
            else:
                
                #unn = (u[i][j-1] - 2u[i][j] + u[i][j+1]) / dxi**2
                append(pc, i, j, 2 * c / J**2 / dxi**2)
                append(pc, i, j+1, - c / J**2 / dxi**2)
                append(pc, i, j-1, - c / J**2 / dxi**2)
                
                #un = (u[i][j+1] - u[i][j-1]) / 2 / dxi
                append(pc, i, j+1, - e / J**2 / 2 / dxi)
                append(pc, i, j-1, e / J**2 / 2 / dxi)


            # For uen
            if i != 0 and j != m-1:          # For left and top - Not generalized!!!

                #uen = (u[i+1][j+1] + u[i-1][j-1] - u[i-1][j+1] - u[i+1][j-1]) / 4 / deta / dxi

                append(pc, i+1, j+1, 2 * b / J**2 / 4 / deta / dxi)
                append(pc, i-1, j-1, 2 * b / J**2 / 4 / deta / dxi)
                append(pc, i-1, j+1, - 2 * b / J**2 / 4 / deta / dxi)
                append(pc, i+1, j-1, - 2 * b / J**2 / 4 / deta / dxi)        


    
def solve(l_new, b_new, h_new, n_new, m_new):

    reset(l_new, b_new, h_new, n_new, m_new)
    
    geoSetup()
    
    fillMat()
    mtx = sparse.coo_matrix((data, (row, col)))
    f = np.ones([neq])
    
    res = linalg.solve(mtx.todense(), f)

    xA = xyGeo[0][0]
    xB = xyGeo[1][0]
    xC = xyGeo[2][0]
    xD = xyGeo[3][0]
    yA = xyGeo[0][1]
    yB = xyGeo[1][1]
    yC = xyGeo[2][1]
    yD = xyGeo[3][1]

    eq = 0
    
    for i in range(0, n):
        for j in range(0, m):
            
            if p[i][j][2] == 2:
                u[i][j] = 0.0
            else:
                u[i][j] = res[eq]
                eq += 1

            eta = p[i][j][0]
            xi = p[i][j][1]
            x[i][j] = xA * (0.5 - eta) * (0.5 + xi) + xB * (0.5 - eta) * (0.5 - xi) + xC * (0.5 + eta) * (0.5 - xi) + xD * (0.5 + eta) * (0.5 + xi)
            y[i][j] = yA * (0.5 - eta) * (0.5 + xi) + yB * (0.5 - eta) * (0.5 - xi) + yC * (0.5 + eta) * (0.5 - xi) + yD * (0.5 + eta) * (0.5 + xi)


def postProc():
    Q = 0.0             # Flow rate through section
    L2 = 0.0
    Linf = 0.0
    xA = xyGeo[0][0]
    xB = xyGeo[1][0]
    xC = xyGeo[2][0]
    xD = xyGeo[3][0]
    yA = xyGeo[0][1]
    yB = xyGeo[1][1]
    yC = xyGeo[2][1]
    yD = xyGeo[3][1]
    for i in range(0, n):
        for j in range(0, m):
            
            #Calculating area
            if i != n-1 and j != m-1:
                eta = p[i][j][0] + deta / 2
                xi = p[i][j][1] + dxi / 2
                xe = (- xA - xB + xC + xD) / 2 + (- xA + xB - xC + xD) * xi
                xn = (xA - xB - xC + xD) / 2 + (- xA + xB - xC + xD) * eta
                ye = (- yA - yB + yC + yD) / 2 + (- yA + yB - yC + yD) * xi
                yn = (yA - yB - yC + yD) / 2 + (- yA + yB - yC + yD) * eta
                Q += (u[i][j] + u[i+1][j] + u[i][j+1] + u[i+1][j+1]) * (xe*yn + xn*ye)

            #Calculating L2 and Linf norm by interpolation to course grid
            
            if i == n-1 and j != m-1:
                
                for l in range(0, int(80 / (m-1))):
                    xi = p[i][j][1] + l * dxi / int(80 / (m-1))
                    uInt = (u[i][j] * (dxi - l * dxi / int(80 / (m-1))) + u[i][j+1] * (l * dxi / int(80 / (m-1)))) / dxi
                    L2 += uInt**2
                    if abs(uInt) > Linf:
                        Linf = uInt
                    
            elif j == m-1 and i != n-1:
                
                for k in range(0, int(80 / (n-1))):
                    eta = p[i][j][0] + k * deta / int(80 / (n-1))
                    uInt = (u[i][j] * (deta - k * deta / int(80 / (n-1))) + u[i+1][j] * (k * deta / int(80 / (n-1)))) / deta
                    L2 += uInt**2
                    if abs(uInt) > Linf:
                        Linf = uInt
                    
            elif i == n-1 and j == m-1:
                
                uInt = u[i][j]
                L2 += uInt**2
                if abs(uInt) > Linf:
                    Linf = uInt
                
            else:
                
                for k in range(0, int(80 / (n-1))):
                    for l in range(0, int(80 / (m-1))):
                        eta = p[i][j][0] + k * deta / int(80 / (n-1))
                        xi = p[i][j][1] + l * dxi / int(80 / (m-1))
                        uInt = (u[i][j] * (deta - k * deta / int(80 / (n-1))) * (dxi - l * dxi / int(80 / (m-1))) +
                                u[i+1][j] * (k * deta / int(80 / (n-1))) * (dxi - l * dxi / int(80 / (m-1))) +
                                u[i][j+1] * (deta - k * deta / int(80 / (n-1))) * (l * dxi / int(80 / (m-1))) +
                                u[i+1][j+1] * (k * deta / int(80 / (n-1))) * (l * dxi / int(80 / (m-1)))
                                ) / deta / dxi
                        L2 += uInt**2
                        if abs(uInt) > Linf:
                            Linf = uInt

    Q *= deta * dxi / 4
    L2 = math.sqrt(L2)
    return Q, L2, Linf


def reset(l_new, b_new, h_new, n_new, m_new):
    global l, b, h, n, m, neq, p, row, col, data, xyGeo, deta, dxi, u, x, y
    l = l_new
    b = b_new
    h = h_new
    n = n_new
    m = m_new
    neq = 0
    p = np.zeros([n, m, 4])
    row = []
    col = []
    data = []
    xyGeo = []
    deta = 1 / (n-1)
    dxi = 1 / (m-1)
    u = np.zeros([n, m])
    x = np.zeros([n, m])
    y = np.zeros([n, m])
               


inp = np.zeros([3])
Q = np.zeros([3])
L2 = np.zeros([3])
Linf = np.zeros([3])
solve(3.0, 1.0, 1.0, 81, 81)     # l, b, h, n, m
Q_ref, L2_ref, Linf_ref = postProc()
for t in range(0, 3):
    n_test = 2**t*10+1
    solve(3.0, 1.0, 1.0, n_test, n_test)     # l, b, h, n, m
    Q_test, L2_test, Linf_test = postProc()
    inp[t] = n_test
    Q[t] = abs((Q_test - Q_ref) / Q_ref) * 100
    L2[t] = abs((L2_test - L2_ref) / L2_ref) * 100
    Linf[t] = abs((Linf_test - Linf_ref) / Linf_ref) * 100

f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(inp, Q, '-o')
axarr[0].set_title('Error % in Flow Rate')
axarr[1].plot(inp, L2, '-o')
axarr[1].set_title('Error % in L2 norm')
axarr[2].plot(inp, Linf, '-o')
axarr[2].set_title('Error % in Linf norm')
plt.xlabel('Grid Size')
plt.show()
