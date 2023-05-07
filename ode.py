import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from mpl_toolkits.mplot3d import Axes3D

f = lambda x, r: r+x**2
tt = np.linspace(-5, 5, 100)
plt.plot(tt, f(tt,-1))
plt.grid()
plt.xlabel('x')
plt.ylabel(r'$\dot{x}$', rotation=0)
plt.show()