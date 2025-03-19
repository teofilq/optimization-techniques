import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

img_orig = plt.imread("imag.png")[:, :, 0]
img_corr = plt.imread("imag_corrupted.png")[:, :, 0]
rows, cols = img_orig.shape

known = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        known[i, j] = 1 if img_corr[i, j] == img_orig[i, j] else 0

U = cp.Variable((rows, cols))
obj = cp.Minimize(cp.tv(U))
constraints = [cp.multiply(known, U) == cp.multiply(known, img_corr)]
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.SCS, verbose=True)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_orig, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(img_corr, cmap='gray')
plt.title("Corrupted Image")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(U.value, cmap='gray')
plt.title("Denoised Image")
plt.axis('off')
plt.show()
