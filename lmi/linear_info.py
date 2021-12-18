#!/usr/bin/env python
import sys
import time
import numpy as np


def frame(t, X):
    """ Returns the matrix X*X.T for a frame at time t """
    time = np.reshape(X[t], (X.shape[1], 1))

    return np.matmul(time, time.T)


def main():
    """ Fuction to compute correlation matrix using Linear Mutual Information  method """

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
            LMI = 0.5 * (
                np.log(np.linalg.det(Ci))
                + np.log(np.linalg.det(Cj))
                - np.log(np.linalg.det(Cij))
            )

            # Generalized Correlation Coeficient
            r = (abs(1 - np.exp((-2 * LMI) / 3))) ** 0.5

            corr_matrix[N, M] = r

    np.save("corr_matrix_linear_info.npy", corr_matrix)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
