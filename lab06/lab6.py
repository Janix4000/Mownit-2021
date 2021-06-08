# %%
from numpy import float32, float64, log2
import sys
from matplotlib import pyplot as plt
import math
import numpy as np
import random
import scipy
import mpmath as mp
from mpmath import mpf, nstr, nprint
# %matplotlib notebook

print("Wersja Pythona:")
print(sys.version)
print(f"Wersja numpy: {np.__version__}")
print("Konfiguracja liczb zmiennoprzecinkowych")
print(sys.float_info)

f64 = np.float64
f32 = np.float32

# %%


def solve_gauss_jordan(M, Y):
    n = M.shape[0]
    M = np.copy(M).astype('float64')
    X = np.copy(Y).astype('float64')
    for i in range(n):
        #         print(X)
        for j in range(i, n):
            if M[j, i] != 0:
                M[j, :], M[i, :] = M[i, :], M[j, :]
                X[j, :], X[i, :] = X[i, :], X[j, :]
                break
        X[i, :] = X[i, :] / M[i, i]
        M[i, :] = M[i, :] / M[i, i]
        for j in range(0, n):
            if j == i:
                continue
            X[j, :] = X[j, :] - (X[i, :] / M[i, i] * M[j, i])
            M[j, :] = M[j, :] - (M[i, :] / M[i, i] * M[j, i])
    return X


# %%
M = np.array([
    [3, 2, 3],
    [3, 2, 1],
    [5, 6, 1]
])

Y = np.array([1, 2, 3])
Y.shape = (-1, 1)

print('----------')
print(f'{Y.flatten()=}')
print(f'{solve_gauss_jordan(M, Y).flatten()=}')
print(f'{np.linalg.solve(M, Y).flatten()=}')
print(np.allclose(np.linalg.solve(M, Y) - solve_gauss_jordan(M, Y), 0))
# %%
