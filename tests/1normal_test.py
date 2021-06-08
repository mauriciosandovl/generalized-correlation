#!/usr/bin/env python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kde_vect import ent_kde_1

def real_ent(x):
    ent = 0.5 * np.log((2*np.pi*np.e) * var)
    return ent

def norm_dist(x, mean, var):
    return (np.pi*var) * np.exp(-0.5*((x-mean)/var)**2)

def gen_sample(n):
    rng = np.random.default_rng(2021)
    sample = []
    while n > 0:
        u = rng.uniform(-3, 3, 1)[0] # from [-1,3]
        e = norm_dist(u, mean, var)
        t = rng.uniform(0, maxtarget, 1)[0] # from [-1,3]
        if e > t:
           n -= 1 
           sample.append(u)
    return np.array(sample)


sns.set_theme(style="darkgrid")

# Parameters for normal distribution
var = .8
dim = 1
nframes = 1000
mean = np.zeros(dim)

smpsize = 100
x = np.linspace(-3, 3, smpsize)
y = np.array([norm_dist(i, mean, var) for i in x])

maxtarget = y.max()
generated = gen_sample(1000)
generated = np.array(generated).reshape(-1, 1)

# Generate normal distributed data
#X = np.random.normal(mean, var, nframes).reshape(-1, 1)

print('Real entropy: ', round(real_ent(generated), 4))
print('KDE entropy:  ', round(ent_kde_1(generated, -3, 3)[0], 4))

# Plots 
#sns.displot(X, kind="kde", rug=True)
#plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
ax1.plot(x, y, color='r')
ax2.hist(generated, range=(-3, 3), density=True)
plt.show()

