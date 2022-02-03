#!/usr/bin/env python
import sys
import time
import numpy as np
import multiprocessing
from   multiprocessing   import sharedctypes
from   scipy.integrate   import dblquad
from   sklearn.neighbors import KDTree, KernelDensity

def get_norm(x):
    """ Función devuelve la norma euclidiana de la posición de los átomos"""
    norm = np.array([np.linalg.norm(i) for j in x for i in j])
    norm = norm.reshape((x.shape[0], x.shape[1], 1))

    return norm


def count_knn(args):

    def mut_inf(x, y):
        px = np.exp(kdeX.score_samples([[x]]))
        py = np.exp(kdeX.score_samples([[y]]))
        pxy = np.exp(kdeXY.score_samples([[x, y]]))
        
        return pxy * np.log(pxy / (px * py))

    def bw(X, dim):
        """ Computes optimal bandwidth """
        # Interquantile range
        q75X, q25X = np.percentile(X, [75, 25])
        iqrX = q75X - q25X
        # Bandwidth as Silverman
        bw = (4 / (nframes * (dim + 4))) ** (1 / (dim + 4))
        
        return bw
    
    N, M = args
    X = data[:, N]
    Y = data[:, M]
    XY = np.hstack((X, Y))
    tmp = np.ctypeslib.as_array(shared_array)

    # Definimos el modelo y ajustamos a los datos
    kdeX = KernelDensity(kernel='gaussian', bandwidth=bw(X, 1)).fit(X)
    kdeY = KernelDensity(kernel='gaussian', bandwidth=bw(Y, 1)).fit(Y)
    kdeXY = KernelDensity(kernel='gaussian', bandwidth=bw(XY, 2)).fit(XY)

    # Evaluate Mutual Information
    bound1 = -3
    bound2 = 3
    x1, x2 = bound1, bound2
    y1, y2 = lambda x: bound1, lambda x: bound2
    
    mi = dblquad(mut_inf, x1, x2, y1, y2)[0]

    tmp[N, M] = max(mi, 0)

    
#-----------------------------------------------------------------------------

start_time = time.time()

# Carga de los datos originales
data = np.load('trj_displacement.npy') #np.load(str(sys.argv[1]))
data = get_norm(data)
nframes = data.shape[0] # Número de frames o conformaciones 
natoms = data.shape[1] # Número de átomos

#-----------------------------------------------------------------------------

# Arguments of get_norm 
list_of_pairs = [ (N, M) for N in range(natoms) for M in range(N) ]

# Initialize the correlation matrix
corr_matrix = np.ctypeslib.as_ctypes(np.zeros((natoms, natoms)))

# Define interable to apply parallel function over norm_data
shared_array = sharedctypes.RawArray(corr_matrix._type_, corr_matrix)

# Apply parallel map of the function in the given array
p2 = multiprocessing.Pool() 
p2.map(count_knn, list_of_pairs)

# Return the map into a n dimensional array
corr_matrix = np.ctypeslib.as_array(shared_array)

#-----------------------------------------------------------------------------

#Aplicamos el coeficiente de correlación generalizado
knn = corr_matrix
knn = (1. - np.exp(-2. * corr_matrix)) ** 0.5

# Guardamos la matriz resultante en un archivo .npy
np.save('1d_multi_kde_matrix.npy', knn)

print("--- %s seconds ---" % (time.time() - start_time) )
