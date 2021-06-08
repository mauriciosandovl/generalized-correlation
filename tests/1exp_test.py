#!/usr/bin/env python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kde_vect import ent_kde_1

def real_ent(x):
    ent = 1 - np.log(lamb)
    return ent


sns.set_theme(style="darkgrid")

# Parameters for normal distribution
lamb = 1
nframes = 1000

# Generate normal distributed data
X = np.random.exponential(lamb, nframes).reshape(-1, 1)

print('Real entropy: ', round(real_ent(X), 4))
print('KDE entropy:  ', round(ent_kde_1(X, 0, 8)[0], 4))

# Plots 
sns.displot(X, kind="kde", rug=True)
#sns.jointplot(x=X[:, 0], y=X[:, 1], kind="kde")
plt.show()
