#!/usr/bin/env python

"""
Module to evaluate the correlation matrix of the trajectory of a protein using
the Pearson correlation coeficient
"""

import sys
import numpy as np
from utils import timeit

INPUT_PATH = str(sys.argv[1])
OUTPUT_PATH = "../outputs/corr_matrix_pearson.npy"


@timeit
def main():
    """Fuction to compute correlation matrix using Pearson method"""

    data = np.load(INPUT_PATH)

    # Number of atoms
    num_atoms = data.shape[1]

    corr_matrix = np.zeros((num_atoms, num_atoms))

    for row in range(num_atoms):
        # Compute only diagonal inferior matrix
        for col in range(row):
            # Variables with all conformations of a pair of atoms
            vect_x = data[:, row]
            vect_y = data[:, col]

            # Mean vectors from the total of frames
            inner_xy = np.diag(np.inner(vect_x, vect_y))
            inner_xx = np.diag(np.inner(vect_x, vect_x))
            inner_yy = np.diag(np.inner(vect_y, vect_y))

            # Pearson correlation coeficient
            corr = np.mean(inner_xy) / (
                np.sqrt(np.mean(inner_xx)) * np.sqrt(np.mean(inner_yy))
            )

            corr_matrix[row, col] = abs(corr)

    np.save(file=OUTPUT_PATH, arr=corr_matrix)


if __name__ == "__main__":
    main()
