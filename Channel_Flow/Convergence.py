import ChannelFlow as cf
import numpy as np
import matplotlib.pyplot as plt

inp = np.zeros([3])
Q = np.zeros([3])
L2 = np.zeros([3])
Linf = np.zeros([3])
cf.solve(3.0, 0.5, 1.0, 81, 81)     # l, b, h, n, m
Q_ref, L2_ref, Linf_ref = cf.postProc()
for t in range(0, 3):
    n_test = 2**t*10+1
    cf.solve(3.0, 0.5, 1.0, n_test, n_test)     # l, b, h, n, m
    Q_test, L2_test, Linf_test = cf.postProc()
    inp[t] = n_test
    Q[t] = abs((Q_test - Q_ref) / Q_ref) * 100
    L2[t] = abs((L2_test - L2_ref) / L2_ref) * 100
    Linf[t] = abs((Linf_test - Linf_ref) / Linf_ref) * 100

f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(inp, Q, '-o')
axarr[0].set_title('l = 3.0, b = 0.5, h = 1.0')
axarr[1].plot(inp, L2, '-o')
axarr[2].plot(inp, Linf, '-o')
plt.xlabel('Grid Size')
plt.show()
