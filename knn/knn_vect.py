#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors, KDTree, KernelDensity

def mi_knn(data, N, M, n_neighbors=3):
    """ 
    Realiza la estimacion de la informacion mutua a partir del conteo del numero
    de atomos nx y ny en la vecindad del k-esimo vecino mas cercano descrito en
    Kraskov
    """
    X = data[:, N]
    Y = data[:, M]
    XY = np.hstack((X, Y))

    # Define the model and fit data. Chebyshev metric corresponds to infinity norm
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
    nbrs.fit(XY)

    # Evaluate the distance to the k-nearest neighbor for each point
    distances, _ = nbrs.kneighbors(XY)
    kth_nbr_dist = distances[:, -1]

    # Initialize variables to count the number of atoms in neighborhood
    nx = np.array([])
    ny = np.array([])

    # Iterate over all points and make the account
    for i, p in enumerate(XY):
        nbr_dist = kth_nbr_dist[i]
        nx_p = 0
        ny_p = 0
        for q in XY:
            xnorm = np.linalg.norm(np.array(p[:3]) - np.array(q[:3]), ord=np.inf)
            ynorm = np.linalg.norm(np.array(p[3:]) - np.array(q[3:]), ord=np.inf)
            if xnorm < nbr_dist:
                nx_p += 1

            elif ynorm < nbr_dist:
                ny_p += 1

        nx = np.append(nx, nx_p)
        ny = np.append(ny, ny_p)

    # Kraskov equation for estimation mutual information
    mi = base_info_level - np.mean(digamma(nx + 1) + digamma(ny + 1))

    return max(mi, 0)


start_time = time.time()

# Load original data
data = np.load('trj_displacement.npy')
nframes = data.shape[0] # Number of frames
natoms = data.shape[1] # Number of atoms

# Parameter of k-nearest neighbor
k = int(3)

base_info_level = digamma(nframes) + digamma(k)

# Initialize correlation matrix
corr_matrix = np.zeros((natoms, natoms))

# Evaluate matrix pair by pair
for N in range(natoms):
    for M in range(N):  # Inferior diagonal matrix
        corr_matrix[N, M] = mi_knn(data, N, M, n_neighbors=k)

# Apply generalized correlation coeficient
corr_matrix = (1. - np.exp(-2. / 3.  * corr_matrix)) ** 0.5

# Save matrix on .npy file
np.save('vect_knn_matrix.npy', corr_matrix)

print('Seconds: ', time.time() - start_time)
