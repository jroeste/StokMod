import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

n=10000 #Number of years
t=365;mu=-2;sigma=1;alpha=0.001;beta=0.95;
C=[0]*n

def simulate_days(days):
	z = 0;
	for t in range(0,days):
		#lam = 3;
		lam = 2 + np.cos(t*np.pi/182.5);
		N = np.random.poisson(lam=lam, size=1);
		z += np.exp(-alpha*t)*sum(np.random.lognormal(mean=mu, sigma=sigma, size=N)); 
	return z;
	
for i in range(0,n):
	C[i] = simulate_days(t); #Sample the insurance amount claims from the lognormal distribution.

mu = np.mean(C);
sigma = np.std(C);
print("Expected value: ", np.mean(C))
print("Variance: " ,np.var(C))


numbers, bins, patches = plt.hist(C, 100, facecolor='green', alpha=0.75);
pdf = st.norm.pdf(bins, mu, sigma);
l = plt.plot(bins, pdf, 'r--', linewidth=1);

def sample_confidence_tail(beta, numbers, bins):
	i = 0;
	number = 0;
	n = sum(numbers);
	print("NUMBA: ", n);
	while (number + numbers[i] < beta*n and i < n):
		number += numbers[i];
		i += 1;
	return bins[i];

bank = sample_confidence_tail(beta, numbers, bins);
print("Bank: ", bank);

plt.show()