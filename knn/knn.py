#!/usr/bin/env python

"""
Module to evaluate the correlation matrix of the trajectory of a protein using
the k-nearest neighbor method
"""

import sys
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from utils import timeit

NUM_NEIGHBORS = int(3)
INPUT_PATH = str(sys.argv[1])
OUTPUT_PATH = f"./corr_matrix_{sys.argv[0][2:-3]}.npy"


def mi_knn(data, row, col, base_info, n_neighbors):
    """Evaluate the mutual information from the number of atoms nx and ny in the
    neighborhood of the k-nearst neighborhood as developed by Kraskov
    """
    vect_x = data[:, row]
    vect_y = data[:, col]
    vect_xy = np.hstack((vect_x, vect_y))

    # Define the model and fit data. Chebyshev metric corresponds to infinity norm
    n_neighbors = NUM_NEIGHBORS
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, metric="chebyshev")
    nbrs.fit(vect_xy)

    # Evaluate the distance to the k-nearest neighbor for each point
    distances, _ = nbrs.kneighbors(vect_xy)
    kth_nbr_dist = distances[:, -1]

    # Initialize variables to count the number of atoms in neighborhood
    nx = np.array([])
    ny = np.array([])

    for i, p in enumerate(vect_xy):
        nbr_dist = kth_nbr_dist[i]
        nx_p = 0
        ny_p = 0
        for q in vect_xy:
            xnorm = np.linalg.norm(np.array(p[:3]) - np.array(q[:3]), ord=np.inf)
            ynorm = np.linalg.norm(np.array(p[3:]) - np.array(q[3:]), ord=np.inf)
            if xnorm < nbr_dist:
                nx_p += 1

            elif ynorm < nbr_dist:
                ny_p += 1

        nx = np.append(nx, nx_p)
        ny = np.append(ny, ny_p)

    # Kraskov equation for estimating mutual information
    mutual_info = base_info - np.mean(digamma(nx + 1) + digamma(ny + 1))

    return max(mutual_info, 0)


@timeit
def main():
    """Fuction to compute correlation matrix using k-nearest neighbor method"""

    data = np.load(INPUT_PATH)

    num_frames = data.shape[0]
    num_atoms = data.shape[1]

    base_info = digamma(num_frames) + digamma(NUM_NEIGHBORS)

    corr_matrix = np.zeros((num_atoms, num_atoms))

    for row in range(num_atoms):
        # Compute only inferior diagonal matrix
        for col in range(row):
            corr_matrix[row, col] = mi_knn(data, row, col, base_info)

    # Generalized correlation coeficient
    corr_matrix = (1.0 - np.exp(-2.0 / 3.0 * corr_matrix)) ** 0.5

    np.save(file=OUTPUT_PATH, arr=corr_matrix)


if __name__ == "__main__":
    main()
