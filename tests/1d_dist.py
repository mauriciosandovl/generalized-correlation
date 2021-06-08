#!/usr/bin/env python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kde_vect import ent_kde_1

def norm_dist(x, mean, var):
    return (np.pi*var) * np.exp(-0.5*((x-mean)/var)**2)

def log_dist(x, s, mu):
    exp = np.exp(-(x-mu) / s) 
    return exp / (s*(1 + exp)**2)

def exp_dist(x, lamb):
    return lamb * np.exp(-lamb * x)

def par_dist(x, x_m, alpha):
    return (alpha * x_m ** alpha) / (x ** (alpha + 1))

def gen_sample(n):
    rng = np.random.default_rng(2021)
    sample = []
    while n > 0:
        u = rng.uniform(-3, 3, 1)[0] # from [-1,3]
        e = norm_dist(u, mean, var)
        t = rng.uniform(0, 3, 1)[0] # from [-1,3]
        if e > t:
           n -= 1
           sample.append(u)
    return np.array(sample)


sns.set_theme(style="darkgrid")

dim = 1
nframes = 1000


# 1. Normal distribution
var = .8
mean = 0

X1 = np.random.normal(mean, var, nframes).reshape(-1, 1)
#X1 = gen_sample(nframes).reshape(-1, 1)

norm_ent = 0.5 * np.log((2*np.pi*np.e) * var)
norm_ent = round(norm_ent, 4)
kde_norm = round(ent_kde_1(X1, -3, 3)[0], 4)

print(f'Normal distribution with mean {mean} and variance {var}')
print('Real entropy: ', norm_ent)
print('KDE entropy:  ', kde_norm)


# 2. Logistic distribution
s = 2
mu = 2

X2 = np.random.logistic(mu, s, nframes).reshape(-1, 1)

log_ent = np.log(s) + 2
log_ent = round(log_ent, 4)
kde_log = round(ent_kde_1(X2, -10, 15)[0], 4)

print(f'\nLogistic distribution with location {mu} and scale {s}')
print('Real entropy: ', log_ent)
print('KDE entropy:  ', kde_log)


# Exponential distribution
lamb = 1

X3 = np.random.exponential(lamb, nframes).reshape(-1, 1)

exp_ent = 1 - np.log(lamb)
exp_ent = round(exp_ent, 4)
kde_exp = round(ent_kde_1(X3, 0, 8)[0], 4)

print(f'\nExponential distribution with rate {lamb}')
print('Real entropy: ', exp_ent)
print('KDE entropy:  ', kde_exp)


# Pareto distribution
x_m = 3
alpha = 2

X4 = ((np.random.pareto(alpha, nframes) + 2) * x_m).reshape(-1, 1)

par_ent = np.log((x_m/alpha) * np.exp(1 + 1/alpha))
par_ent = round(par_ent, 4)
kde_par = round(ent_kde_1(X4, 0, 50)[0], 4)

print(f'\nPareto distribution with scale {x_m} and shape {alpha}')
print('Real entropy: ', par_ent)
print('KDE entropy:  ', kde_par)


# Plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

sns.histplot(X1, stat='density',  kde=True, ax=ax1, legend=False)
ax1.set_title("Normal Distribution")
s1 = f'Real ent: {norm_ent}\n KDE ent: {kde_norm}' 
ax1.text(0.60, 0.95, s1, transform=ax1.transAxes, fontsize=9,
    verticalalignment='top', bbox=props)

sns.histplot(X2, stat='density', kde=True, ax=ax2, legend=False)
ax2.set_title("Logistic Distribution")
s2 = f'Real ent: {log_ent}\n KDE ent: {kde_log}' 
ax2.text(0.60, 0.95, s2, transform=ax2.transAxes, fontsize=9,
    verticalalignment='top', bbox=props)

sns.histplot(X3, stat='density', kde=True, ax=ax3, legend=False)
ax3.set_title("Exponential Distribution")
s3 = f'Real ent: {exp_ent}\n KDE ent: {kde_exp}' 
ax3.text(0.60, 0.95, s3, transform=ax3.transAxes, fontsize=9,
    verticalalignment='top', bbox=props)

sns.histplot(X4, stat='density', kde=True, ax=ax4, legend=False)
ax4.set_title("Pareto Distribution")
s4 = f'Real ent: {par_ent}\n KDE ent: {kde_par}' 
ax4.text(0.60, 0.95, s4, transform=ax4.transAxes, fontsize=9,
    verticalalignment='top', bbox=props)

plt.show()

