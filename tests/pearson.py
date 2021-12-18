#!/usr/bin/env python
import numpy as np

def pearson(data, N, M):
    nframes = data.shape[0]
    # Creamos las variables X y Y con todas las 401 conformaciones de los 
    # átomos X y Y
    X = data[:, N].reshape(-1)
    Y = data[:, M].reshape(-1)

    # Calculamos los vectores promedio del total de frames
    XX = np.diag(np.inner(X, X))
    YY = np.diag(np.inner(Y, Y))
    XY = np.diag(np.inner(X, Y))

    # Ecuación 1. Coeficiente de correlación de Pearson  
    r =  np.mean(XY) / (np.sqrt(np.mean(XX)) * np.sqrt(np.mean(YY)))

    return abs(r)

