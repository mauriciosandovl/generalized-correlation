#!/usr/bin/env python
import sys
import time
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def get_norm(x):
    """ Evalua la norma euclidiana de la posición de los átomos """
    norm = np.array([np.linalg.norm(i) for j in x for i in j])
    norm = norm.reshape((x.shape[0], x.shape[1], 1))

    return norm


def mi_kde(data, N, M):
    # Vectores con las 401 configuraciones de los átomos N y M
    X = data[:, N]
    Y = data[:, M]
    XY = np.hstack((X, Y))

    # Cuantiles del 25 y 75 por ciento para X, Y y XY
    q75X, q25X = np.percentile(X, [75 ,25])
    q75Y, q25Y = np.percentile(Y, [75 ,25])
    q75XY, q25XY = np.percentile(XY, [75 ,25])
    
    # Rango intercuantil
    iqrX = q75X - q25X
    iqrY = q75Y - q25Y
    iqrXY = q75XY - q25XY
    
    # Bandwidth de Silverman con la modificacion de Steuer
    bw1 = min(np.std(X), iqrX / 1.34) * (4 / (3 * nframes)) ** 0.2
    bw2 = min(np.std(Y), iqrY / 1.34) * (4 / (3 * nframes)) ** 0.2
    bw3 = min(np.std(XY), iqrXY / 1.34) * (nframes ** (-1/6) )    

    # Definimos el modelo y ajustamos a los datos
    kdeX = KernelDensity(kernel='gaussian', bandwidth=bw1).fit(X)
    kdeY = KernelDensity(kernel='gaussian', bandwidth=bw2).fit(Y)
    kdeXY = KernelDensity(kernel='gaussian', bandwidth=bw3).fit(XY)

    # Evaluate Mutual Information
    mi = 0
    for u, v in XY:
        px = np.exp(kdeX.score_samples([[u]]))
        py = np.exp(kdeY.score_samples([[v]]))
        pxy = np.exp(kdeXY.score_samples([[u, v]]))
        
        #mi += pxy * np.log( pxy / ( px * py ) )
        mi += np.log(pxy) - np.log(px) - np.log(py)
        
    mi /= XY.shape[0]

    return max(mi, 0)


def corr_matrix(data):
    """ 
    Calcula la matriz de correlacion para un metodo dado y 
    guarda el resultado en un archivo .npy
    """
    natoms = data.shape[1] # Número de átomos
    dim = data.shape[2] # Dimension of data
    
    # Inicializamos la matriz de correlacion
    corr_matrix = np.zeros((natoms, natoms))
    
    # Evaluamos la matrix entrada a entrada
    for N in range(natoms):
        for M in range(N):  # Matriz diagonal inferior
            corr_matrix[N, M] = mi_kde(data, N, M)

    # Aplicamos el coeficiente de correlacion generalizado de Lange
    corr_matrix = (1 - np.exp(-2 / dim * corr_matrix)) ** 0.5

    # Guardamos la matriz resultante en un archivo .npy
    np.save('matrix.npy', corr_matrix)


start_time = time.time()

# Carga de los datos originales
data = np.load('trj_displacement.npy') # str(sys.argv[1])
norm_data = get_norm(data) # lo sacamos de la funcion para hacerlo solo una vez.
nframes = data.shape[0] # Número de frames o conformaciones 
natoms = data.shape[1] # Número de átomos

# Compute correlation matrix
corr_matrix(norm_data)


print("--- %s seconds ---" % (time.time() - start_time) )

