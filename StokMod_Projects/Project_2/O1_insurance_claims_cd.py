import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

#n=10000 #Number of years
#t=365;mu=-2;sigma=1;alpha=0.001;beta=0.95;


def simulate_days(days, mu, sigma, lam, discount):
	n = 0;
	z = 0;
	for t in range(0,days):
		l = lam(t);
		tmp = np.random.poisson(lam=l, size=1);
		n += tmp;
		z += discount(t)*sum(np.random.lognormal(mean=mu, sigma=sigma, size=tmp)); 
	return n, z;

def simulate_n(n, days, mu, sigma, lam, discount):
	N=[0]*n
	C=[0]*n
	for i in range(0,n):
		N[i], C[i] = simulate_days(days, mu, sigma, lam, discount); #Sample the insurance amount claims from the lognormal distribution.
	return N, C;


def sample_confidence_tail(beta, numbers, bins):
	i = 0;
	number = 0;
	n = sum(numbers);
	while (number + numbers[i] < beta*n and i < n):
		number += numbers[i];
		i += 1;
	return bins[i];


def constants():
	n = 1000;
	t = 365;
	mu = -2;
	sigma = 1;
	alpha = 0.001;
	return n, t, mu, sigma, alpha;

def simulate_constant_intensity():
	n, t, mu, sigma, alpha = constants();
	lam = lambda t: 3;
	discount = lambda t: 1;
	N, C = simulate_n(n, t, mu, sigma, lam, discount);
	plot_histogram(C);
	
	
def simulate_varying_intensity():
	n, t, mu, sigma, alpha = constants();
	lam = lambda t: 2 + np.cos(t*np.pi/182.5);
	discount = lambda t: 1;
	N, C = simulate_n(n, t, mu, sigma, lam, discount);
	plot_histogram(C);
			
def simulate_constant_intensity_discounted():	
	n, t, mu, sigma, alpha = constants();
	lam = lambda t: 3;
	discount = lambda t: np.exp(-alpha*t);
	N, C = simulate_n(n, t, mu, sigma, lam, discount);
	plot_histogram(C);

def simulate_varying_intensity_discounted():	
	n, t, mu, sigma, alpha = constants();
	lam = lambda t: 2 + np.cos(t*np.pi/182.5);
	discount = lambda t: np.exp(-alpha*t);
	N, C = simulate_n(n, t, mu, sigma, lam, discount);
	plot_histogram(C);
	
def plot_histogram(X):
	numbers, bins, patches = plt.hist(X, 100, facecolor='blue', alpha=0.75);
	print("Expected value: ", np.mean(X));
	print("Variance: ", np.var(X));
	plt.show();

simulate_constant_intensity();	
simulate_varying_intensity();
simulate_constant_intensity_discounted();	
simulate_varying_intensity_discounted();
