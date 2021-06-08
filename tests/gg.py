#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def some_function(v):
    return v**3 - 2*v**2 - 0.5*v + 4


def gen_sample(n):
    rng = np.random.default_rng(2021)
    sample = []
    while n > 0:
        u = rng.uniform(-1, 3, 1)[0] # from [-1,3]
        e = some_function(u)
        t = rng.uniform(0, maxtarget, 1)[0] # from [-1,3]
        if e > t:
           n -= 1 
           sample.append(u)
    return sample


smpsize = 100
x = np.linspace(-1, 3, smpsize)
y = np.array([some_function(i) for i in x])

maxtarget = y.max()
generated = gen_sample(1000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
ax1.plot(x, y, color='r')
ax2.hist(generated, bins=100, range=(-1, 3), density=True)
plt.show()

