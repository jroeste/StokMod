import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt
import random as rnd
import scipy.stats as st 

# Creating a mean vector and covariance matrix conditional on a datablock x_B
# Input parameters

# Constructs distance matrices
def h(ta, tb):
	na = len(ta);
	nb = len(tb);	
	onesa = np.ones(na);
	onesb = np.ones(nb);
	return abs(np.outer(ta, onesb) - np.outer(onesa, tb));


def sample(tmin, tmax, nrPoints, avg, var):
	n = nrPoints;
	mu = avg;
	sigma = var;

	t = np.linspace(tmin, tmax, n);
	H = h(t, t);
	S = sigma**2 * np.exp(-phi*H); # 
	L = np.linalg.cholesky(S); #Cholesky factorize covariance matrix S, to yield L.

	z = np.random.normal(0, 1, n);
	return mu + L*z;

# Returns Matern correlation function
def corrMatern(h,sigma,phi):
	return (sigma**2)*(1+phi*h)*np.exp(-phi*h)

# Given to parameter blocks, function constructs distance and covariance matrices
def matrixConstructor(t_A, t_B, sigma, phi):
	H_A = h(t_A, t_A);
	H_B = h(t_B, t_B);
	H_AB = h(t_A, t_B);
	S_A = corrMatern(H_A,sigma,phi);
	S_B = corrMatern(H_B,sigma,phi);
	S_AB = corrMatern(H_AB,sigma,phi);	
	return S_A, S_B, S_AB

# Given an array of expected values, variances and a limit; function returns probability of element i being greater than probLim
def qualProb(expVal, Var, probLim):
	n = len(expVal);
	qualDiff = probLim*np.ones(n)-expVal;
	qualProbs = np.zeros(n);
	for i in range(0,n):
		qualProbs[i] = 1 - st.norm.cdf(qualDiff[i]/Var[i]);
	return qualProbs;


def G1():
	# Assigment specific parameters
	sigma = 4;
	phi_m = 0.2;
	t_A = np.linspace(10,80,141);
	t_B = [19.4, 29.7, 36.1, 50.7, 71.9];
	x_B = [50.1, 39.1, 54.7, 42.1, 40.9];
	mu_A = 50*np.ones(141);
	mu_B = 50*np.ones(len(x_B));

	# Construct distance, and covariance matrices
	S_A, S_B, S_AB = matrixConstructor(t_A, t_B, sigma, phi_m);

	# Calculate conditional expected valuea and variance
	mult1 = np.matmul(S_AB,np.linalg.inv(S_B));
	mult2 = np.matmul(mult1,x_B-mu_B);
	condExpValue = mu_A + mult2
	mult3 = np.matmul(S_AB,np.linalg.inv(S_B))
	mult4 = np.matmul(mult3,np.matrix.transpose(S_AB))
	condVariance = S_A - mult4;

	# Calculate 90% conditional prediction interval. 
	n = len(condExpValue)
	predIntervall = np.zeros((2,n));
	z_val = 1.645; # Sample from Z-distribution that yields 90% prediction interval.
	for i in range(0,n):
		predIntervall[0,i] = condExpValue[i] + z_val*math.sqrt(condVariance[i,i]);
		predIntervall[1,i] = condExpValue[i] - z_val*math.sqrt(condVariance[i,i]);

	# Plot conditional expected value with 90 % prediction interval.
	plt.plot(t_A, condExpValue, label='Expected value')
	plt.plot(t_B, x_B, 'bs', label='Data block B')
	plt.plot(t_A, predIntervall[0,:], 'r--', t_A, predIntervall[1,:], 'r--', label='90% prediction interval')
	plt.xlabel("t")
	plt.ylabel("x(t)")
	plt.title("Expected x_A, at instances t_A. Conditional on (t_B, x_B(t))")
	plt.legend()
	plt.grid()
	plt.show()

def G2():
	# Assigment specific parameters
	sigma = 4;
	phi_m = 0.2;
	t_A = np.linspace(10,80,141);
	t_B = [19.4, 29.7, 36.1, 50.7, 71.9];
	x_B = [50.1, 39.1, 54.7, 42.1, 40.9];
	mu_A = 50*np.ones(141);
	mu_B = 50*np.ones(len(x_B));

	# Construct distance, and covariance matrices
	S_A, S_B, S_AB = matrixConstructor(t_A, t_B, sigma, phi_m);

	# Calculate conditional expected valuea and variance
	mult1 = np.matmul(S_AB,np.linalg.inv(S_B));
	mult2 = np.matmul(mult1,x_B-mu_B);
	condExpValue = mu_A + mult2
	mult3 = np.matmul(S_AB,np.linalg.inv(S_B))
	mult4 = np.matmul(mult3,np.matrix.transpose(S_AB))
	condVariance = S_A - mult4;

	# Calculate the probability of quality being greater than 57 for a certain temp
	qualProbs = qualProb(condExpValue, condVariance, 57);
	plt.plot(t_A, qualProbs, label='Probability')
	plt.xlabel("t")
	plt.ylabel("Probability")
	plt.title("Probability of x(t) > 57. Conditional on (t_B, x_B(t))")
	plt.legend()
	plt.grid()
	plt.show()


def G3():
	# Assignment specific parameters
	sigma = 4;
	phi_m = 0.2;
	t_A = np.linspace(10,80,141);
	t_B = [19.4, 29.7, 36.1, 40.7, 50.7, 71.9];
	x_B = [50.1, 39.1, 54.7, 49.7, 42.1, 40.9];
	mu_A = 50*np.ones(141);
	mu_B = 50*np.ones(len(x_B));

	# Construct distance, and covariance matrices
	S_A, S_B, S_AB = matrixConstructor(t_A, t_B, sigma, phi_m);

	# Calculate conditional expected valuea and variance
	mult1 = np.matmul(S_AB,np.linalg.inv(S_B));
	mult2 = np.matmul(mult1,x_B-mu_B);
	condExpValue = mu_A + mult2
	mult3 = np.matmul(S_AB,np.linalg.inv(S_B))
	mult4 = np.matmul(mult3,np.matrix.transpose(S_AB))
	condVariance = S_A - mult4;

	# Calculate 90% conditional prediction interval. 
	n = len(condExpValue)
	predIntervall = np.zeros((2,n));
	z_val = 1.645; # Sample from Z-distribution that yields 90% prediction interval.
	for i in range(0,n):
		predIntervall[0,i] = condExpValue[i] + z_val*math.sqrt(condVariance[i,i]);
		predIntervall[1,i] = condExpValue[i] - z_val*math.sqrt(condVariance[i,i]);

	# Plot conditional expected value with 90 % prediction interval.
	plt.plot(t_A, condExpValue, label='Expected value')
	plt.plot(t_B, x_B, 'bs', label='Data block B')
	plt.plot(t_A, predIntervall[0,:], 'r--', t_A, predIntervall[1,:], 'r--', label='90% prediction interval')
	plt.xlabel("t")
	plt.ylabel("x(t)")
	plt.title("Expected x_A, at instances t_A. Conditional on (t_B, x_B(t))")
	plt.legend()
	plt.grid()
	plt.show()



G2()