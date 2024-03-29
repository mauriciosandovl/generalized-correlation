#!/usr/bin/env python

"""
Module to evaluate the correlation matrix of the trajectory of a protein using
the k-nearest neighbor method
"""

import sys
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from utils import timeit, gen_corr_coef

NUM_NEIGHBORS = int(3)
INPUT_PATH = str(sys.argv[1])
OUTPUT_PATH = "/home/mauricio/generalized-correlation/outputs/corr_matrix_knn.npy"


@timeit
def main():
    """Fuction to compute correlation matrix using k-nearest neighbor method"""

    data = np.load(INPUT_PATH)

    num_atoms = data.shape[1]

    corr_matrix = np.zeros((num_atoms, num_atoms))

    for row in range(num_atoms):
        # Compute only inferior diagonal matrix
        for col in range(row):
            corr_matrix[row, col] = mi_knn(data, row, col, NUM_NEIGHBORS)

    # Generalized correlation coeficient
    corr_matrix = gen_corr_coef(corr_matrix)

    np.save(file=OUTPUT_PATH, arr=corr_matrix)


def mi_knn(data, row, col, n_neighbors=3):
    """Evaluate the mutual information from the number of atoms nx and ny in the
    neighborhood of the k-nearst neighborhood as developed by Kraskov
    """

    num_frames = data.shape[0]

    vect_x = data[:, row]
    vect_y = data[:, col]
    vect_xy = np.hstack((vect_x, vect_y))

    # Define the model and fit data. Chebyshev metric corresponds to infinity norm
    n_neighbors = NUM_NEIGHBORS
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="chebyshev")
    nbrs.fit(vect_xy)

    # Evaluate the distance to the k-nearest neighbor for each point
    distances, _ = nbrs.kneighbors(vect_xy)
    kth_nbr_dist = distances[:, -1]

    # Initialize variables to count the number of atoms in neighborhood
    nbrs_x = np.array([])
    nbrs_y = np.array([])

    for i, p in enumerate(vect_xy):
        nbr_dist = kth_nbr_dist[i]
        nbrs_x_p, nbrs_y_p = 0, 0
        for q in vect_xy:
            xnorm = np.linalg.norm(np.array(p[:3]) - np.array(q[:3]), ord=np.inf)
            ynorm = np.linalg.norm(np.array(p[3:]) - np.array(q[3:]), ord=np.inf)
            if xnorm < nbr_dist:
                nbrs_x_p += 1

            elif ynorm < nbr_dist:
                nbrs_y_p += 1

        nbrs_x = np.append(nbrs_x, nbrs_x_p)
        nbrs_y = np.append(nbrs_y, nbrs_y_p)

    # Kraskov equation for estimating mutual information
    base_info = digamma(num_frames) + digamma(n_neighbors)
    mutual_info = base_info - np.mean(digamma(nbrs_x + 1) + digamma(nbrs_y + 1))

    return max(mutual_info, 0)


if __name__ == "__main__":
    main()
