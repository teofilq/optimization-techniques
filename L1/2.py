import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

res = minimize(f, [0, 0], method='Nelder-Mead')
print("Punctul de minim găsit:", res.x)

xgrid = np.linspace(-2, 2, 400)
ygrid = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(xgrid, ygrid)
Z = 100 * (Y - X ** 2) ** 2 + (1 - X) ** 2

plt.figure()
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20))
plt.plot(res.x[0], res.x[1], 'ro', label='Minim')
plt.title("Graficul funcției Rosenbrock")
plt.legend()
plt.show()
