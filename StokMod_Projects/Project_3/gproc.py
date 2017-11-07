import numpy as np
import matplotlib.pyplot as plt

def sample(tmin, tmax):
	#tmin = 1;
	#tmax = 100;
	n = tmax - tmin + 1;

	mu = 1;
	sigma = 1;

	t = np.linspace(tmin, tmax, n);
	ones = np.ones(n);
	H = abs(np.outer(t, ones) - np.outer(ones, t));
	S = sigma**2 * np.exp(-phi*H);
	L = np.linalg.cholesky(S);

	z = np.random.normal(0, 1, n);
	return mu + L*z;

def h(ta, tb):
	#t = np.linspace(tmin, tmax, n);
	na = len(ta);
	nb = len(tb);
	
	onesa = np.ones(na);
	onesb = np.ones(nb);
	return abs(np.outer(ta, onesb) - np.outer(onesa, tb));

def samplemodel(t, phi):
	mu = 0;
	sigma = 1;

	H = h(t, t);
	S = sigma**2 * np.exp(-phi*H);
	L = np.linalg.cholesky(S);

	z = np.random.normal(0, 1, n);
	return mu + L*z;
	

def main():
	phi = 3/10;
	tmin = 1;
	tmax = 100;
	t = np.linspace(tmin, tmax, tmax - tmin + 1);
	for i in range(10):
		z = sample(t, phi);
		plt.hist(z);
		