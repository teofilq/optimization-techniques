import numpy as np
from scipy.optimize import linprog
import cvxpy as cp

np.random.seed(42)

m = 3
n = 4

a = np.random.randint(20, 50, size=m)
total_supply = a.sum()
b = np.random.randint(10, 30, size=n)
b = b / b.sum() * total_supply


C = np.random.randint(1, 10, size=(m, n))

c = C.flatten()

A_eq = []
b_eq = []
for i in range(m):
    row = np.zeros(m * n)
    for j in range(n):
        row[i * n + j] = 1
    A_eq.append(row)
    b_eq.append(a[i])
for j in range(n):
    row = np.zeros(m * n)
    for i in range(m):
        row[i * n + j] = 1
    A_eq.append(row)
    b_eq.append(b[j])

A_eq = np.array(A_eq)
b_eq = np.array(b_eq)
bounds = [(0, None)] * (m * n)

res_linprog = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
sol_linprog = res_linprog.x.reshape(m, n) if res_linprog.success else None

x_cvx = cp.Variable(m * n, nonneg=True)
objective = cp.Minimize(cp.matmul(c, x_cvx))
constraints = [cp.matmul(A_eq, x_cvx) == b_eq]
prob = cp.Problem(objective, constraints)
result_cvx = prob.solve(solver=cp.SCS)
sol_cvx = x_cvx.value.reshape(m, n)

print(sol_linprog)
print(sol_cvx)
