#!/usr/bin/env python

"""
Module to evaluate the correlation matrix of the 1-dimensional trajectory of a
protein using the kernel density estimation method
"""

import sys
import numpy as np
from sklearn.neighbors import KernelDensity
from utils import timeit, get_norm, gen_corr_coef

INPUT_PATH = str(sys.argv[1])
OUTPUT_PATH = "../outputs/corr_matrix_kde_1d.npy"


@timeit
def main():
    """Fuction to compute 1-dimensional correlation matrix using kernel density
    estimation method
    """

    data = np.load(INPUT_PATH)
    data = get_norm(data)

    num_atoms = data.shape[1]

    corr_matrix = np.zeros((num_atoms, num_atoms))

    for row in range(num_atoms):
        # Compute only inferior diagonal matrix
        for col in range(row):
            corr_matrix[row, col] = mi_kde(data, row, col)
            print(row, col, corr_matrix[row, col])

    corr_matrix = gen_corr_coef(corr_matrix, dim=1)

    np.save(file=OUTPUT_PATH, arr=corr_matrix)


def mi_kde(data, row, col):
    """Evaluate the mutual information by estimating the density of the vectors
    in the trajectory of the protein
    """

    def opt_bw(vect):
        """Computes optimal bandwidth"""
        num_frames = vect.shape[0]
        dim = vect.shape[1]
        # Interquantile range
        q75, q25 = np.percentile(vect, [75, 25])
        iqr = q75 - q25
        # Bandwidth as Silverman with Steuer modification
        return min(np.std(vect), iqr / 1.34) * (
            (4 / (num_frames * (dim + 2))) ** (1 / (dim + 4))
        )

    num_frames = data.shape[0]

    vect_x = data[:, row]
    vect_y = data[:, col]
    vect_xy = np.hstack((vect_x, vect_y))

    # Define the model and fit data using optimal bandwidth
    kde_x = KernelDensity(kernel="gaussian", bandwidth=opt_bw(vect_x)).fit(vect_x)
    kde_y = KernelDensity(kernel="gaussian", bandwidth=opt_bw(vect_y)).fit(vect_y)
    kde_xy = KernelDensity(kernel="gaussian", bandwidth=opt_bw(vect_xy)).fit(vect_xy)

    # Evaluate Mutual Information
    mutual_info = 0
    for val_x, val_y in vect_xy:
        dens_x = np.exp(kde_x.score_samples([[val_x]]))
        dens_y = np.exp(kde_y.score_samples([[val_y]]))
        dens_xy = np.exp(kde_xy.score_samples([[val_x, val_y]]))

        mutual_info += dens_xy * np.log(dens_xy / (dens_x * dens_y))
        # mutual_info += (np.log(dens_xy) - np.log(dens_x) - np.log(dens_y))

        mutual_info /= num_frames

    return max(mutual_info, 0)


if __name__ == "__main__":
    main()
