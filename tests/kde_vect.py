#!/usr/bin/env python
import time
import numpy as np
from scipy.integrate import tplquad, dblquad, quad
from sklearn.neighbors import KDTree, KernelDensity

def ent_kde(X, bound):
    """ Computes entropy for a 3-dimensional frame """
    def entropy(x, y, z):
        px = np.exp(kdeX.score_samples([[x, y, z]]))
        
        return -px * np.log(px)

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
    kdeX = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X)

    # Evaluate entropy correctly
    x1, x2 = -bound, bound
    y1, y2 = lambda x: -bound, lambda x: bound
    z1, z2 = lambda x, y: -bound, lambda x, y: bound
    
    return tplquad(entropy, x1, x2, y1, y2, z1, z2)

def ent_kde_2(X, bound):
    """ Computes entropy for a 2-dimensional frame """
    def entropy(x, y):
        px = np.exp(kdeX.score_samples([[x, y]]))
        
        return -px * np.log(px)

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
    kdeX = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X)

    # Evaluate entropy correctly
    x1, x2 = -bound, bound
    y1, y2 = lambda x: -bound, lambda x: bound
    
    return dblquad(entropy, x1, x2, y1, y2)


def ent_kde_1(X, bound1, bound2):
    """ Computes entropy for a 1-dimensional frame """
    def entropy(x):
        px = np.exp(kdeX.score_samples([[x]]))
        
        return -px * np.log(px)

    # Data parameters
    nframes = X.shape[0]
    dim = 1
    
    # Interquantile range
    q75X, q25X = np.percentile(X, [75, 25])
    iqrX = q75X - q25X
    # Bandwidth as Silverman
    mnm = min(np.std(X), iqrX / 1.34)
    bw = mnm * (4 / (nframes * (dim + 4))) ** (1 / (dim + 4))
        
    # Define the model and fit with data
    kdeX = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X)

    # Evaluate entropy correctly
    x1, x2 = bound1, bound2
    
    return quad(entropy, x1, x2)

