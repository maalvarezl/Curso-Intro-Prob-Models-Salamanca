import numpy as np
import matplotlib.pyplot as plt

# Fix the seed
seedn = 10**4
np.random.seed(seedn)

N_tot = 100
sigma2 = 0.01
beta = 1/sigma2
betaInv = 1/beta
x_tot = np.linspace(0, 1, num=N_tot) 
y_tot = np.sin(2*np.pi*x_tot)
t_tot = y_tot + np.sqrt(betaInv)*np.random.randn(N_tot)

# Choose 60% for traininig 
N = 10
index = np.random.permutation(N_tot)
t = t_tot[index[0:N]]
x = x_tot[index[0:N]]
t_test = t_tot[index[N:N_tot]]
x_test = x_tot[index[N:N_tot]]

fig, ax = plt.subplots()

ax.plot(x_tot, y_tot, 'b', x, t, '.r', linewidth=2.0)
plt.show()

# Let's compute the basis functions 
M = 10 # Number of basis functions
mu = np.linspace(0, 1, M-1)
aux = np.diff(mu)
s2 = 0.05*aux[0]
PHI_tot = np.zeros((N_tot, M))
PHI_tot[:, 0] = np.ones(N_tot)

for i in range(1, M):
    PHI_tot[:, i] = np.exp(-(x_tot-mu[i-1])**2/(2*s2))

fig, ax = plt.subplots()

ax.plot(x_tot, PHI_tot, linewidth=2.0)
plt.show()

# Compute the basis functions only for the training data
PHI = np.zeros((N, M))
PHI[:, 0] = np.ones(N)

for i in range(1, M):
    PHI[:, i] = np.exp(-(x-mu[i-1])**2/(2*s2))

# We compute the maximum-likelihood solution
wML = np.linalg.solve(PHI.T@PHI, PHI.T@t[:, np.newaxis])

# Compute the prediction for all the inputs
y_hat = PHI_tot@wML

fig, ax = plt.subplots()

ax.plot(x_tot, y_tot, 'b', x_tot, y_hat, 'k', linewidth=2.0)
plt.show()

