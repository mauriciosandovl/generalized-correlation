#!/usr/bin/env python
import sys
import time
import numpy as np

def frame(t, X):
    """ Returns the matrix X*X.T for a frame at time t """
    tiempo = np.reshape(X[t], (X.shape[1], 1))

    return np.matmul(tiempo, tiempo.T)


start_time = time.time()

data = np.load('trj_displacement.npy')#np.load(str(sys.argv[1]))
nframes = data.shape[0] # Nuber of frames
natoms = data.shape[1] # Number of atoms

# Initialize correlation matrix
corr_matrix = np.zeros((natoms, natoms))

for N in range(natoms):
    for M in range(N):  # Inferior diagonal matrix

        X = data[:, N]
        Y = data[:, M]

        alt_xy = np.array([np.concatenate((X[i], Y[i])) for i in range(len(X))])
        
        # Evaluate correlation matrix
        A = frame(0, X)
        B = frame(0, Y)
        C = frame(0, alt_xy)

        for i in range(1, nframes):
            A = A + frame(i, X)
            B = B + frame(i, Y)
            C = C + frame(i, alt_xy)

        Ci = A / nframes
        Cj = B / nframes 
        Cij = C / nframes

        # Equation for Linear Mutual Information
        LMI = 0.5 * ( np.log(np.linalg.det(Ci)) + np.log(np.linalg.det(Cj)) 
                      - np.log(np.linalg.det(Cij)) )

        # Generalized Correlation Coeficient
        r = (abs(1 - np.exp((-2 * LMI) / 3))) ** 0.5 

        corr_matrix[N, M] = r


# Guardamos la matriz resultante en un archivo .npy
np.save('linear_info_matrix.npy', corr_matrix)

print('--- %s seconds ---' % (time.time() - start_time))
