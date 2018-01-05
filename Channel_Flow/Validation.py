import ChannelFlow as cf
import numpy as np
import matplotlib.pyplot as plt


cf.solve(3.0, 0.5, 1.0, 21, 21)
plt.figure(1)
plt.contourf(cf.x, cf.y, cf.u)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.text(0.1, 1.03, 'l = 3.0, b = 0.5, h = 1.0, N = 21')

cf.solve(3.0, 0.5, 1.0, 41, 41)
plt.figure(2)
plt.contourf(cf.x, cf.y, cf.u)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.text(0.1, 1.03, 'l = 3.0, b = 0.5, h = 1.0, N = 41')
plt.show()
