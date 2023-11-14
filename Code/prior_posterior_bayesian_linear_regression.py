import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Fix the seed
seedn = 10**3
np.random.seed(seedn)

w0t = -0.3
w1t = 0.5
betaInv = 0.01;
beta = 1/betaInv;
alpha = 2;
alphaInv = 1/alpha;

# Generate data
N_tot = 100;
x_tot = np.linspace(-1, 1, num=N_tot)
true_y = w0t + w1t*x_tot
noisy_y = true_y + np.sqrt(betaInv)*np.random.randn(N_tot)
# Choose a subset of the noisy data
N = 20
index = np.random.permutation(N_tot)
t = noisy_y[index[0:N]]
x = x_tot[index[0:N]]

fig, ax = plt.subplots()

ax.plot(x_tot, true_y, 'r', x, t, 'ob', linewidth=2.0)
ax.set_title('DATA')
plt.show()

# Plot the prior distribution

Nt = 50
w0 = np.linspace(-1, 1, num=Nt)
w1 = np.linspace(-1, 1, num=Nt)
W0, W1 = np.meshgrid(w0, w1)

w0v = np.reshape(W0, (Nt*Nt, 1), 'F')
w1v = np.reshape(W1, (Nt*Nt, 1), 'F')
mean_prior = np.zeros(2)
cov_prior = alpha*np.eye(2)

def plot_gaussian(mean_local, cov_local, w0, w1, w0t, w1t, w0v, w1v,  
                                                  Nt, stitle):
    pdf_vals = multivariate_normal.pdf(np.column_stack((w0v, w1v)), \
                        mean = mean_local, cov=cov_local)
    fig, ax = plt.subplots()
    plt.contourf(w0, w1, np.reshape(pdf_vals, (Nt, Nt), 'F'))
    plt.axis('scaled')
    plt.colorbar()
    plt.plot(w0t, w1t, 'xk', linewidth=5)
    ax.set_title(stitle)
    plt.show()

stitle = 'Gaussian prior'
plot_gaussian(mean_prior, cov_prior, w0, w1, w0t, w1t, w0v, w1v, Nt, stitle)

# Plot sample lines from the parameters sampled from the Gaussian
def plot_samples(mean_local, cov_local, Nt, nsamples, stitle, pdata=False):
    Y = np.zeros((N_tot, nsamples))
    samples = multivariate_normal.rvs(mean_local, cov_local, nsamples)
    for i in range(nsamples):
        c, m = samples[i, :]    
        Y[:, i] = c + m*x_tot 
              
    fig, ax = plt.subplots()  
    plt.plot(x_tot, Y, 'r', linewidth=2)
    ax.set_title(stitle)
    if pdata:
        plt.plot(xtr, ttr, 'ob', linewidth=2)
    plt.show()

nsamples = 5
stitle = 'Parameter samples from the prior'
plot_samples(mean_prior, cov_prior, Nt, nsamples, stitle)

# Pick a random data point and plot tha Gaussian again
xtr = x[15]
ttr = t[15]
PHI = np.array([1, xtr])
PHIv = PHI[np.newaxis, :]
SNinv = alpha*np.eye(2) + beta*PHIv.T@PHIv
SN = np.linalg.solve(SNinv, np.eye(2))
mN = beta*(SN@PHIv.T)*ttr

# Now we plot the Gaussian posterior after one point
stitle = 'Gaussian posterior after one observation'
plot_gaussian(mN[:, -1], SN, w0, w1, w0t, w1t, w0v, w1v, Nt, stitle)
# Now we plot sample lines from the parameters sampled from the Gaussian when
# we have observed one instance
stitle = 'Parameter samples from the posterior after one obs'
plot_samples(mN[:, -1], SN, Nt, nsamples, stitle, pdata=True)

# Pick another random data point and plot tha Gaussian again (N=2)
N_local = 2
xtr = np.array((x[15], x[10]))
ttr = np.array((t[15], t[10]))

def plots_ng2(N_local, xtr, ttr, alpha, beta):
    PHI = np.zeros((N_local,2))
    PHI[:, 0] = np.ones(N_local)
    PHI[:, 1] = xtr
    SNinv = alpha*np.eye(2) + beta*PHI.T@PHI
    SN = np.linalg.solve(SNinv, np.eye(2))
    mN = beta*(SN@PHI.T@ttr[:, np.newaxis])
    # Now we plot the Gaussian posterior after one point
    stitle = 'Gaussian posterior after '+str(N_local)+' observations'
    plot_gaussian(mN[:, -1], SN, w0, w1, w0t, w1t, w0v, w1v, Nt, stitle)
    # Now we plot sample lines from the parameters sampled from the Gaussian when
    # we have observed one instance
    stitle = 'Parameter samples from the posterior after '+str(N_local)+' obs'
    plot_samples(mN[:, -1], SN, Nt, nsamples, stitle, pdata=True)

plots_ng2(N_local, xtr, ttr, alpha, beta)

# Pick five random data point and plot tha Gaussian again

N_local = 5
xtr = np.array((x[15], x[10], x[5], x[0], x[8]))
ttr = np.array((t[15], t[10], t[5], t[0], t[8]))
plots_ng2(N_local, xtr, ttr, alpha, beta)
