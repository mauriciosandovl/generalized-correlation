#!/usr/bin/env python
import sys
import time
import numpy as np

start_time = time.time()

data = np.load('trj_displacement_new.npy')#np.load(str(sys.argv[1]))
nframes = data.shape[0] # Número de frames o conformaciones 
natoms = data.shape[1] # Número de átomos

# Inicializamos la matriz de correlación
corr_matrix = np.zeros((natoms, natoms))

for N in range(natoms):
    for M in range(N):  # Matriz diagonal inferior
        # Creamos las variables X y Y con todas las 401 conformaciones de los 
        # átomos X y Y
        X = data[:, N]
        Y = data[:, M]

        # Calculamos los vectores promedio del total de frames
        XY = np.diag(np.inner(X, Y))
        XX = np.diag(np.inner(X, X))
        YY = np.diag(np.inner(Y, Y))

        # Ecuación 1. Coeficiente de correlación de Pearson  
        r =  np.mean(XY) / (np.sqrt(np.mean(XX)) * np.sqrt(np.mean(YY)))

        corr_matrix[N, M] = abs(r)


# Guardamos la matriz resultante en un archivo .npy
np.save('pearson_matrix_new.npy', corr_matrix)

print('--- %s seconds ---' % (time.time() - start_time))
