#!/usr/bin/env python

"""
Module to evaluate the correlation matrix of the trajectory of a protein using
the linear mutual information
"""

import sys
import numpy as np
from utils import timeit

INPUT_PATH = str(sys.argv[1])
OUTPUT_PATH = f"./corr_matrix_{sys.argv[0][2:-3]}.npy"


def frame_eval(time, frame):
    """Returns the matrix frame*frame.T for a frame at time t"""
    time_frame = np.reshape(frame[time], (frame.shape[1], 1))

    return np.matmul(time_frame, time_frame.T)


@timeit
def main():
    """Fuction to compute correlation matrix using Linear Mutual Information method"""

    data = np.load(INPUT_PATH)

    # Number of frames or conformations
    nframes = data.shape[0]
    # Number of atoms
    natoms = data.shape[1]

    corr_matrix = np.zeros((natoms, natoms))

    for row in range(natoms):
        # Compute only diagonal inferior matrix
        for col in range(row):
            # Variables with all conformations of a pair of atoms
            vect_x = data[:, row]
            vect_y = data[:, col]
            alt_xy = np.array(
                [np.concatenate((vect_x[i], vect_y[i])) for i in range(len(vect_x))]
            )

            # Evaluate correlation matrix
            c_i, c_j, c_ij = 0, 0, 0

            for i in range(nframes):
                c_i += frame_eval(i, vect_x)
                c_j += frame_eval(i, vect_y)
                c_ij += frame_eval(i, alt_xy)

            c_i = c_i / nframes
            c_j = c_j / nframes
            c_ij = c_ij / nframes

            # Equation for Linear Mutual Information
            lmi = 0.5 * (
                np.log(np.linalg.det(c_i))
                + np.log(np.linalg.det(c_j))
                - np.log(np.linalg.det(c_ij))
            )

            # Generalized correlation coeficient
            corr = (abs(1 - np.exp((-2 * lmi) / 3))) ** 0.5

            corr_matrix[row, col] = corr

    np.save(file=OUTPUT_PATH, arr=corr_matrix)


if __name__ == "__main__":
    main()
