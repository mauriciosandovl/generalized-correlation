#!/usr/bin/env python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kde_vect import ent_kde_2

def real_ent(x):
    det = np.linalg.det(np.array(cov))
    ent = 0.5 * np.log((2*np.pi*np.e)**dim * det)
    #ent = (.5 * dim) + (.5 * dim * np.log(2*np.pi)) + (.5 * np.log(det))

    return ent


sns.set_theme(style="darkgrid")

# Parameters for normal distribution
r = .8
dim = 2
nframes = 1000
mean = np.zeros(dim)
cov = np.array([
    [1, r],
    [r, 1]]
)

# Generate normal distributed data
X = np.random.multivariate_normal(mean, cov, nframes)

print('Real entropy: ', round(real_ent(X), 4))
print('KDE entropy:  ', round(ent_kde_2(X, 3)[0], 4))

# Plots 
#sns.displot(x=X[:, 0], y=X[:, 1], kind="kde", rug=True)
sns.jointplot(x=X[:, 0], y=X[:, 1], kind="kde")
plt.show()
