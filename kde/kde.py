#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree, KernelDensity

def mi_kde_vect(data, N, M):
    def bw(X, dim):
        """Computes optimal bandwidth"""
        # Interquantile range
        q75X, q25X = np.percentile(X, [75, 25])
        iqrX = q75X - q25X
        # Bandwidth as Silverman
        mnm = min(np.std(X), iqrX / 1.34)
        bw = mnm * (4 / (nframes * (dim + 4))) ** (1 / (dim + 4))
        
        return bw
    
    nframes = data.shape[0] # Número de frames o conformaciones 
    
    # Vectores con las 401 configuraciones de los átomos N y M
    X = data[:, N]
    Y = data[:, M]
    XY = np.hstack((X, Y))

    # Definimos el modelo y ajustamos a los datos
    kdeX = KernelDensity(kernel='gaussian', bandwidth=bw(X, 3)).fit(X)
    kdeY = KernelDensity(kernel='gaussian', bandwidth=bw(Y, 3)).fit(Y)
    kdeXY = KernelDensity(kernel='gaussian', bandwidth=bw(XY, 6)).fit(XY)

    # Evaluate Mutual Information
    mi = 0
    for x1, x2, x3, y1, y2, y3 in XY:
        px = np.exp(kdeX.score_samples([[x1, x2, x3]]))
        py = np.exp(kdeY.score_samples([[y1, y2, y3]]))
        pxy = np.exp(kdeXY.score_samples([[x1, x2, x3, y1, y2, y3]]))

        mi += pxy * np.log( pxy / ( px * py ) )
        #mi += (np.log(pxy) - np.log(px) - np.log(py))

    mi /= nframes

    return max(mi, 0)


start_time = time.time()

# Carga de los datos originales
data = np.load('trj_displacement.npy')
nframes = data.shape[0] # Número de frames o conformaciones 
natoms = data.shape[1] # Número de átomos

# Inicializamos la matriz de correlacion
corr_matrix = np.zeros((natoms, natoms))

# Evaluamos la matrix entrada a entrada
for N in range(natoms):
    for M in range(N):  # Matriz diagonal inferior
        corr_matrix[N, M] = mi_kde_vect(data, N, M)

# Aplicamos el coeficiente de correlacion generalizado de Lange
corr_matrix = (1 - np.exp(-2 / 3 * corr_matrix)) ** 0.5

# Guardamos la matriz resultante en un archivo .npy
np.save('vect_kde_matrix.npy', corr_matrix)

print("--- %s seconds ---" % (time.time() - start_time))
