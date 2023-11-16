import numpy as np
import matplotlib.pyplot as plt

# Fix the seed
seedn = 10**4
np.random.seed(seedn)

N_tot = 100
sigma2 = 0.01
beta = 1/sigma2
betaInv = 1/beta
x_tot = np.linspace(0, 1, num=N_tot)[:, np.newaxis] 
y_tot = np.sin(2*np.pi*x_tot)
noise = np.sqrt(betaInv)*np.random.randn(N_tot)[:, np.newaxis]
t_tot = y_tot + noise 

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

def rbfKernel(x, xp, l, sf):
    aux = x**2 - 2*x@xp.T + (xp.T**2)
    K = sf*np.exp(-aux/(2*l**2))
    return K

# Compute the covariance matrix
l = 0.1
sf = 0.5
#K = rbfKernel_loop(x, xp, l, sf)
xp = x
K = rbfKernel(x, xp, l, sf)

L = np.linalg.cholesky(K + betaInv*np.eye(N))
Linv = np.linalg.solve(L, np.eye(N))
alpha = (Linv.T@Linv)@t
Ktest = rbfKernel(x_tot, x, l, sf)
mpred = Ktest@alpha
Ktest_test = rbfKernel(x_tot, x_tot, l, sf)
covpred = Ktest_test - Ktest@(Linv.T@Linv)@Ktest.T + betaInv*np.eye(N_tot)
vpred = np.diag(covpred)
# Compute the predictive distribution 

fig, ax = plt.subplots()
ax.plot(x_tot, y_tot, 'b', x_tot, mpred, 'k', linewidth=2.0)
ax.plot(x_tot, mpred+2*np.sqrt(vpred[:, np.newaxis]), 'k--', linewidth=2.0)
ax.plot(x_tot, mpred-2*np.sqrt(vpred[:, np.newaxis]), 'k--', linewidth=2.0)
ax.plot(x, t, 'or')
plt.show()







