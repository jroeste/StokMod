__author__ = 'julie'
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

def compute_marginal_probability(P, x0, n):
	dim = len(x0);

	x = np.zeros((dim, n));
	
	x[:,0] = x0;
	for i in range(0, n - 1):
		x[:,i + 1] = np.matmul(P, x[:,i]);
	return x;

def sample(p):
	s = rnd.rand();
	p_sample = [s, 1 - s];
	return np.less(p_sample, p).astype(np.float64);

def draw_realization(P, x0, n):
	dim = len(x0);
	r = np.zeros((dim, n));

	r[:,0] = sample(x0);
	for i in range(0, n - 1):
		p = np.matmul(P, r[:,i]);
		r[:,i + 1] = sample(p);
	return r;

def draw_realizations(P, x0, n, r):
	dim = len(x0);
	x = np.zeros((r, n));

	for i in range(1, r):
		x[i,:] = draw_realization(P, x0, n)[0];
	return x;

def test_marginal_probability():
	P = [[0.95, 0.00], [0.05, 1]];
	x0 = [0.99, 0.01];

	n = 50;
	x = compute_marginal_probability(P, x0, n);
	#plt.plot(range(0,n), x[1,:]);
	#plt.savefig("teest.png");
	#print(x);

def test_realizations():
	P = [[0.95, 0.00], [0.05, 1]];
	x0 = [0.99, 0.01];

	n = 50;
	r = 25;

	x = draw_realizations(P, x0, n, r);
	plt.matshow(x, cmap="hot");
	plt.show();

def test_cost():
	P = [[0.95, 0.00], [0.05, 1]];
	x0 = [0.99, 0.01];

	n = 50;
	x = compute_marginal_probability(P, x0, n);
	
	repair_cost = 5000;

	individual_cost = repair_cost*sum(x[1,:]);
	all_cost = 100000;
	cost = min(individual_cost, all_cost);
	print(individual_cost);
	print(all_cost);

test_marginal_probability();
test_realizations();
test_cost();
