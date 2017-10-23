import numpy as np
import matplotlib.pyplot as plt

# Save figures
SAVEFIG = 0;

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


def sample_confidence_tail(C, beta):
	hist, bins = np.histogram(C, 100);
	i = 0;
	number = 0;
	n = sum(hist);
	while (number + hist[i] < beta*n and i < n):
		number += hist[i];
		i += 1;
	bank = bins[i];
	print('The insurance company must have', bank, 'in order to be 95% confident they will have enough.')
	return bins[i];


def constants():
	n = 1000;
	t = 365;
	mu = -2;
	sigma = 1;
	alpha = 0.001;
	beta = 0.95;
	return n, t, mu, sigma, alpha, beta;

def simulate_constant_intensity():
	n, t, mu, sigma, alpha, beta = constants();
	lam = lambda t: 3;
	discount = lambda t: 1;
	N, C = simulate_n(n, t, mu, sigma, lam, discount);
	plot_histogram(C, 'constant_intensity.pdf');
	
	
def simulate_varying_intensity():
	n, t, mu, sigma, alpha, beta = constants();
	lam = lambda t: 2 + np.cos(t*np.pi/182.5);
	discount = lambda t: 1;
	N, C = simulate_n(n, t, mu, sigma, lam, discount);
	plot_histogram(C, 'varying_intensity.pdf');
			
def simulate_constant_intensity_discounted():	
	n, t, mu, sigma, alpha, beta = constants();
	lam = lambda t: 3;
	discount = lambda t: np.exp(-alpha*t);
	N, C = simulate_n(n, t, mu, sigma, lam, discount);
	sample_confidence_tail(C, beta);
	plot_histogram(C, 'constant_intensity_discounted.pdf');

def simulate_varying_intensity_discounted():	
	n, t, mu, sigma, alpha, beta = constants();
	lam = lambda t: 2 + np.cos(t*np.pi/182.5);
	discount = lambda t: np.exp(-alpha*t);
	N, C = simulate_n(n, t, mu, sigma, lam, discount);
	sample_confidence_tail(C, beta);
	plot_histogram(C, 'varying_intensity_discounted.pdf');
	
def plot_histogram(X, fname):
	numbers, bins, patches = plt.hist(X, 100, facecolor='blue', alpha=0.75, edgecolor='black');
	print('Expected value: ', np.mean(X));
	print('Variance: ', np.var(X));
	if SAVEFIG:
		plt.savefig(fname);
		plt.clf();
	else:
		plt.title('Histogram of Cost:');
		plt.show();

simulate_constant_intensity(); # C
simulate_varying_intensity(); # C
simulate_constant_intensity_discounted(); # D
simulate_varying_intensity_discounted(); # D
