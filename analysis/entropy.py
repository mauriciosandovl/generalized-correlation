#!/usr/bin/env python
import time
import numpy as np
from scipy.special import digamma
from scipy.integrate import tplquad, dblquad, quad
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors


def ent_kde(X, bound1=0, bound2=1):
    """X.shape(-1, n), n=1,2,3"""
    # Data parameters
    nframes = X.shape[0]
    dim = X.shape[1]

    # Interquantile range
    q75X, q25X = np.percentile(X, [75, 25])
    iqrX = q75X - q25X
    # Bandwidth as Silverman
    mnm = min(np.std(X), iqrX / 1.34)
    bw = mnm * (4 / (nframes * (dim + 4))) ** (1 / (dim + 4))

    # Define the model and fit with data
    kdeX = KernelDensity(kernel="gaussian", bandwidth=bw).fit(X)

    #
    if dim == 1:
        """Computes entropy for a 1-dimensional frame"""

        def entropy(x):
            px = np.exp(kdeX.score_samples([[x]]))

            return -px * np.log(px)

        # Evaluate entropy correctly
        x1, x2 = bound1, bound2

        ent, _ = quad(entropy, x1, x2)

    elif dim == 2:

        def entropy(x, y):
            px = np.exp(kdeX.score_samples([[x, y]]))

            return -px * np.log(px)

        # Evaluate entropy correctly
        x1, x2 = bound1, bound2
        y1, y2 = lambda x: bound1, lambda x: bound2

        ent, _ = dblquad(entropy, x1, x2, y1, y2)

    elif dim == 3:

        def entropy(x, y, z):
            px = np.exp(kdeX.score_samples([[x, y, z]]))

            return -px * np.log(px)

        # Evaluate entropy correctly
        x1, x2 = bound1, bound2
        y1, y2 = lambda x: bound1, lambda x: bound2
        z1, z2 = lambda x, y: bound1, lambda x, y: bound2

        ent, _ = tplquad(entropy, x1, x2, y1, y2, z1, z2)

    return max(ent, 0)


def ent_knn(X, n_neighbors):
    """Realiza la estimacion de la informacion mutua a partir del conteo del
    numero de atomos nx y ny en la vecindad del k-esimo vecino mas cercano
    descrito en Kraskov
    """
    nframes = X.shape[0]
    dim = X.shape[1]

    # Definimos el modelo de knn y ajustamos los datos. En este caso,
    # la metrica de Chebyshev corresponde la norma infinito
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="chebyshev")
    nbrs.fit(X)

    # Calculamos la distancia al k-esimo vecino mas proximo para cada punto
    distances, _ = nbrs.kneighbors(X)
    kth_nbr_dist = distances[:, -1]

    # Variables para el conteo del número de átomos en la vecindad
    nx = np.array([])

    if dim == 1:
        # Iteramos sobre todo el resto de puntos para hacer el conteo
        for i, p in enumerate(X):
            nbr_dist = kth_nbr_dist[i]
            nx_p = 0
            for q in X:
                if abs(q[0] - p[0]) < nbr_dist:
                    nx_p += 1

            nx = np.append(nx, nx_p)

    else:
        # Iteramos sobre todo el resto de puntos para hacer el conteo
        for i, p in enumerate(X):
            nbr_dist = kth_nbr_dist[i]
            nx_p = 0
            for q in X:
                xnorm = np.linalg.norm(np.array(p[:3]) - np.array(q[:3]), ord=np.inf)
                if xnorm < nbr_dist:
                    nx_p += 1

            nx = np.append(nx, nx_p)

    # Ecuación 8 de Kraskov para estimar la informacion mutua
    ent = (
        -1 / nframes * np.sum(digamma(nx + 1))
        + digamma(nframes)
        + dim / nframes * np.sum(np.log(2 * kth_nbr_dist))
    )

    return max(ent, 0)
