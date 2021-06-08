#!/usr/bin/env python
import numpy as np

def pearson(data):
    nframes = data.shape[0]
    # Creamos las variables X y Y con todas las 401 conformaciones de los 
    # átomos X y Y
    X = data[:, 3:]
    Y = data[:, :3]

    # Calculamos los vectores promedio del total de frames
    XY = np.diag(np.inner(X, Y))
    XX = np.diag(np.inner(X, X))
    YY = np.diag(np.inner(Y, Y))

    # Ecuación 1. Coeficiente de correlación de Pearson  
    r =  np.mean(XY) / (np.sqrt(np.mean(XX)) * np.sqrt(np.mean(YY)))

    return abs(r)

