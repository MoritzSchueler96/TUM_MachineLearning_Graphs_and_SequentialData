import itertools

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix

def laplacian(A):
    D = A.sum(axis=0)
    return np.identity(A.shape[0]) * D - A

def format_partitioning(f):
    c1 = (f > 0.0).nonzero()[0]
    c2 = (f < 0.0).nonzero()[0]
    return f"{{{', '.join(map(str, c1))}}} {{{', '.join(map(str, c2))}}}"

edges = np.array([
    (0, 1, 2), (0, 3, 2), (1, 2, 4), (1, 3, 2), (2, 3, 1),
    (2, 4, 3), (2, 5, 4), (2, 6, 4), (4, 5, 2), (5, 6, 1)]).T
A = coo_matrix((edges[2], (edges[0], edges[1])), shape=(7, 7))
A = A.toarray()
A = A + A.T
L = laplacian(A)

# Global minimum cut
fs = [((f := np.array(f_)) @ L @ f / 4, f)
      for f_ in itertools.product([-1, 1], repeat=A.shape[0])
      if len(set(f_)) > 1]
min_cost, min_cut = min(fs, key=lambda f: f[0])
print(f"Global minimum cut is {format_partitioning(min_cut)} at cost {min_cost}")

# Approximate ratio cut
lambda_, v = eigh(L, eigvals=(1, 1))
print(f"Approximate ratio cut is {format_partitioning(v.squeeze())}")

# Approximate normalized cut
D_sqrt_inv = np.diag(1 / np.sqrt(A.sum(axis=1)))
L_normalized = D_sqrt_inv @ L @ D_sqrt_inv
lambda_, v = eigh(L_normalized, eigvals=(1, 1))
print(f"Approximate normalized cut is {format_partitioning(v.squeeze())}")
