import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

img_orig = plt.imread("imag.png")[:, :, 0]
img_corr = plt.imread("imag_corrupted.png")[:, :, 0]
rows, cols = img_orig.shape

rho = 0.1

U = cp.Variable((rows, cols))
obj = cp.Minimize(0.5 * cp.sum_squares(U - img_corr) + 
                  rho * (cp.sum_squares(U[1:, :] - U[:-1, :]) + cp.sum_squares(U[:, 1:] - U[:, :-1])))
prob = cp.Problem(obj)
prob.solve(solver=cp.SCS, verbose=True)

plt.figure()
plt.imshow(U.value, cmap='gray')
plt.title("Denoised Image with Quadratic Fidelity")
plt.axis('off')
plt.show()
