#!/usr/bin/env python
import sys
import time
import numpy as np


def main():
    """ Fuction to compute correlation matrix using Pearson method """
    start_time = time.time()

    data_path = str(sys.argv[1])
    data = np.load(data_path)

    # Number of frames or conformations
    nframes = data.shape[0]
    # Number of atoms
    natoms = data.shape[1]

    corr_matrix = np.zeros((natoms, natoms))

    for N in range(natoms):
        # Compute only diagonal inferior matrix
        for M in range(N):
            # Variables with all conformations of a pair of atoms
            X = data[:, N]
            Y = data[:, M]

            # Mean vectors from the total of frames
            XY = np.diag(np.inner(X, Y))
            XX = np.diag(np.inner(X, X))
            YY = np.diag(np.inner(Y, Y))

            # Pearson correlation coeficient
            r = np.mean(XY) / (np.sqrt(np.mean(XX)) * np.sqrt(np.mean(YY)))

            corr_matrix[N, M] = abs(r)

    np.save("./corr_matrix_pearson.npy", corr_matrix)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()

