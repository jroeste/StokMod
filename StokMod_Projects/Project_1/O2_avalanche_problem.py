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

def draw_realization(P, x0, n):
	dim = len(x0);
	x = np.zeros((dim, n));

	x[:,0] = x0;
	for i in range(0, n - 1):
		tmp = rnd.rand();
		p = [tmp, 1 - tmp];
		x[:,i + 1] = np.less(p, np.matmul(P, x[:,i])).astype(np.float64);
	return x;

def test_marginal_probability():
	P = [[0.95, 0.00], [0.05, 1]];
	x0 = [0.99, 0.01];

	n = 50;
	#x = compute_marginal_probability(P, x0, n);
	#plt.plot(range(0,n), x[1,:]);
	#plt.savefig("teest.png");
	#print(x);
	r = draw_realization(P, x0, n);
	print(r);

test_marginal_probability();
