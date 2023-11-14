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
    
# Variational Bayes solution
niters = 20
E_alpha = 1 # Initial value for the expected value of alpha
E_wT_w = 1 # Initial value for the expected value of wT*w
a0 = 1
b0 = 1
tv = t[:, np.newaxis]
E_alphav = np.zeros(niters)
for i in range(niters):
    # Update the moments for q(alpha)
    aN = a0 + M/2
    bN = b0 + (1/2)*E_wT_w
    # Update the moments for q(w)
    E_alphav[i] = E_alpha    
    E_alpha = aN/bN    
    SNInv = E_alpha*np.eye(M) +  beta*(PHI.T@PHI)
    SN = np.linalg.solve(SNInv, np.eye(M))
    mN = beta*SN@PHI.T@tv;
    E_wT_w = np.trace(mN@mN.T +  SN)

# Plot the convegence of E_alpha
fig, ax = plt.subplots()
ax.plot(E_alphav, 'k', linewidth=2.0)
plt.show()


# Compute the predictive distribution 

mpred = PHI_tot@mN;
vpred = betaInv + np.diag(PHI_tot@SN@PHI_tot.T)
fig, ax = plt.subplots()
ax.plot(x_tot, y_tot, 'b', x_tot, mpred, 'k', linewidth=2.0)
ax.plot(x_tot, mpred+2*np.sqrt(vpred[:, np.newaxis]), 'k--', linewidth=2.0)
ax.plot(x_tot, mpred-2*np.sqrt(vpred[:, np.newaxis]), 'k--', linewidth=2.0)
ax.plot(x, t, 'or')
plt.show()




# Type II maximum-likelihood to estimate alpha and beta
# alpha = 1e-1
# betae = 1e-1
# rho, V = np.linalg.eig(PHI.T@PHI)
# lamb = betae*rho
# niters = 100
# betav = np.zeros(niters)
# alphav = np.zeros(niters)
# betav[0] = betae
# alphav[0] = alpha
# tv = t[:, np.newaxis]
# for n in range(niters):
#     betav[n] = betae
#     alphav[n] = alpha
#     A = alpha*np.eye(M) + betae*(PHI.T@PHI)
#     Ainv = np.linalg.solve(A, np.eye(M))
#     mN = betae*(Ainv@PHI.T@tv)
#     gamma = np.sum(lamb/(lamb+alpha))
#     alpha = gamma/(mN.T@mN)
#     aux = tv - PHI@mN 
#     betaeInv = (aux.T@aux)/(N-gamma)
#     betae = 1/betaeInv
#     lamb = betae*rho

# # We now compute the mean and covariance for the predictive distribution
# SNInv = alpha*np.eye(M) +  betae*(PHI.T@PHI)
# SN = np.linalg.solve(SNInv, np.eye(M))
# mN = betae*SN@PHI.T@tv;



