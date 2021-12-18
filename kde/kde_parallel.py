#!/usr/bin/env python
import sys
import time
import numpy as np
import multiprocessing
from multiprocessing import sharedctypes
from sklearn.neighbors import KDTree, KernelDensity


def count_knn(args):
    def bw(X, dim):
        """ Computes optimal bandwidth """
        # Interquantile range
        q75X, q25X = np.percentile(X, [75, 25])
        iqrX = q75X - q25X
        # Bandwidth as Silverman
        mnm = min(np.std(X), iqrX / 1.34)
        bw = mnm * (4 / (nframes * (dim + 4))) ** (1 / (dim + 4))
        
        return bw
    
    N, M = args
    X = data[:, N]
    Y = data[:, M]
    XY = np.hstack((X, Y))
    tmp = np.ctypeslib.as_array(shared_array)

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

    tmp[N, M] = max(mi, 0)

    
#-----------------------------------------------------------------------------

start_time = time.time()

# Carga de los datos originales
data = np.load('trj_displacement.npy') #np.load(str(sys.argv[1]))
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
knn = (1. - np.exp(-2. / 3. * corr_matrix)) ** 0.5

# Guardamos la matriz resultante en un archivo .npy
np.save('multi_kde_matrix.npy', knn)

print("--- %s seconds ---" % (time.time() - start_time) )
