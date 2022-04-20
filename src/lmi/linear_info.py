#!/usr/bin/env python

"""
Module to evaluate the correlation matrix of the trajectory of a protein using
the linear mutual information
"""

import sys
import numpy as np
from utils import timeit, gen_corr_coef

INPUT_PATH = str(sys.argv[1])
OUTPUT_PATH = "/home/mauricio/generalized-correlation/outputs/corr_matrix_linear_info.npy"


@timeit
def main():
    """Fuction to compute correlation matrix using Linear Mutual Information method"""

    data = np.load(INPUT_PATH)

    num_frames = data.shape[0]
    num_atoms = data.shape[1]

    corr_matrix = np.zeros((num_atoms, num_atoms))

    for row in range(num_atoms):
        # Compute only diagonal inferior matrix
        for col in range(row):
            vect_x = data[:, row]
            vect_y = data[:, col]
            vect_xy = np.hstack((vect_x, vect_y))

            # Evaluate correlation matrix
            c_i, c_j, c_ij = 0, 0, 0

            for i in range(num_frames):
                c_i += frame_eval(i, vect_x)
                c_j += frame_eval(i, vect_y)
                c_ij += frame_eval(i, vect_xy)

            c_i /= num_frames
            c_j /= num_frames
            c_ij /= num_frames

            # Equation for Linear Mutual Information
            lmi = 0.5 * (
                np.log(np.linalg.det(c_i))
                + np.log(np.linalg.det(c_j))
                - np.log(np.linalg.det(c_ij))
            )

            # Generalized correlation coeficient
            corr = gen_corr_coef(lmi)

            corr_matrix[row, col] = corr

    np.save(file=OUTPUT_PATH, arr=corr_matrix)


def frame_eval(time, frame):
    """Returns the matrix frame*frame.T for a frame at time t"""
    time_frame = np.reshape(frame[time], (frame.shape[1], 1))

    return np.matmul(time_frame, time_frame.T)


if __name__ == "__main__":
    main()
