import ChannelFlow as cf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import ConvexHull

arr = np.zeros([110, 2])

h = np.arange(0.1, 1.1, 0.1)
b = np.arange(0.0, 1.1, 0.1)

k = 0
for i in h:
    for j in b:
        cf.solve(3.0, j, i, 21, 21)
        Q, L2, Linf = cf.postProc()
        I = cf.getMOI(0.05)
        arr[k][0] = Q
        arr[k][1] = I
        k += 1

hull = ConvexHull(arr)
plt.plot(arr[hull.vertices,0], arr[hull.vertices,1], 'r--', lw=2)
plt.plot(arr[hull.vertices[0],0], arr[hull.vertices[0],1], 'ro')
plt.xlabel('Flow rate')
plt.ylabel('Moment of Inertia')
plt.title('Convex Hull for Optimization')
plt.show()
