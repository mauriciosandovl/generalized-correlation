#!/usr/bin/env python
import sys
import time
import numpy as np
import multiprocessing
from   multiprocessing   import sharedctypes
from   scipy.special     import digamma
from   sklearn.neighbors import NearestNeighbors

def count_knn(args):
    N, M = args
    X = data[:, N]
    Y = data[:, M]
    XY = np.hstack((X, Y))
    tmp = np.ctypeslib.as_array(shared_array)
    
    # Define the model and fit data. Chebyshev metric corresponds to infinity norm
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
    nbrs.fit(XY)

    # Evaluate the distance to the k-nearest neighbor for each point
    distances, _ = nbrs.kneighbors(XY)
    kth_nbr_dist = distances[:, -1]

    # Initialize variables to count the number of atoms in neghborhood
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

    # Kraskov equation for estimating mutual information
    mi = base_info_level - np.mean(digamma(nx + 1) + digamma(ny + 1))

    tmp[N, M] = max(mi, 0)

    
#-----------------------------------------------------------------------------

start_time = time.time()

# Load original data
data = np.load('trj_displacement.npy') #np.load(str(sys.argv[1]))
nframes = data.shape[0] # Number of frames 
natoms = data.shape[1] # Number of atoms

# Parameter of k-nearest neighbor
k = int(3)

base_info_level = digamma(nframes) + digamma(k)

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
np.save('multi_knn_matrix.npy', knn)

print("--- %s seconds ---" % (time.time() - start_time) )
