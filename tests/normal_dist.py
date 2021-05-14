#!/usr/bin/env python
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance

x, y = np.random.multivariate_normal(mean, cov, 5000).T

print('Pearson Correlation Coefitient: ', pearsonr(x, y), sep='\n')

plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
