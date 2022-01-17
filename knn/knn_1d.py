#!/usr/bin/env python

"""
Module to evaluate the correlation matrix of the 1-dimensional trajectory of a
protein using the k-nearest neighbor method
"""

import sys
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from utils import timeit, get_norm

NUM_NEIGHBORS = int(3)
INPUT_PATH = str(sys.argv[1])
OUTPUT_PATH = f"./corr_matrix_{sys.argv[0][2:-3]}.npy"


def mi_knn(data, N, M, n_neighbors=3):
    """
    Realiza la estimacion de la informacion mutua a partir del conteo del numero
    de atomos nx y ny en la vecindad del k-esimo vecino mas cercano descrito en
    Kraskov
    """
    X = data[:, N]
    Y = data[:, M]
    XY = np.hstack((X, Y))

    # Definimos el modelo de knn y ajustamos los datos. En este caso,
    # la metrica de Chebyshev corresponde la norma infinito
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="chebyshev")
    nbrs.fit(XY)

    # Calculamos la distancia al k-esimo vecino mas proximo para cada punto
    distances, _ = nbrs.kneighbors(XY)
    kth_nbr_dist = distances[:, -1]

    # Variables para el conteo del número de átomos en la vecindad
    nx = np.array([])
    ny = np.array([])

    # Iteramos sobre todo el resto de puntos para hacer el conteo
    for i, p in enumerate(XY):
        nbr_dist = kth_nbr_dist[i]
        nx_p = 0
        ny_p = 0
        for q in XY:
            if abs(q[0] - p[0]) < nbr_dist:
                nx_p += 1

            elif abs(q[1] - p[1]) < nbr_dist:
                ny_p += 1

        nx = np.append(nx, nx_p)
        ny = np.append(ny, ny_p)

    # Ecuación 8 de Kraskov para estimar la informacion mutua
    mi = (
        digamma(n_neighbors)
        - np.mean(digamma(nx + 1) + digamma(ny + 1))
        + digamma(num_frames)
    )

    return max(mi, 0)


def corr_matrix(method):
    # Inicializamos la matriz de correlacion
    corr_matrix = np.zeros((num_atoms, num_atoms))

    # Evaluamos la matrix entrada a entrada
    for N in range(num_atoms):
        for M in range(N):  # Matriz diagonal inferior
            corr_matrix[N, M] = method(norm_data, N, M, n_neighbors=2)

    # Aplicamos el coeficiente de correlacion generalizado de Lange
    corr_matrix = (1 - np.exp(-2 * corr_matrix)) ** 0.5

    # Guardamos la matriz resultante en un archivo .npy
    np.save("matrix.npy", corr_matrix)


start_time = time.time()

# Carga de los datos originales
data = np.load("trj_displacement.npy")
norm_data = get_norm(data)
num_frames = norm_data.shape[0]
num_atoms = norm_data.shape[1]

# Compute correlation matrix
corr_matrix(mi_knn)

print("Seconds: ", time.time() - start_time)
