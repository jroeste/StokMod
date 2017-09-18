__author__ = 'julie'
# -*- coding: utf-8 -*-

import numpy as np

def compute_marginal_probability(P, x0, n):
	dim = len(x0);

	x = np.zeros((dim, n));
	
	x[:,0] = x0;
	for i in range(0, n - 1):
		x[:,i + 1] = np.matmul(P, x[:,i]);
	return x;
	
def test_marginal_probability():
	P = [[0.95, 0.05], [0, 1]];
	x0 = [0.99, 0.01];

	n = 50;
	x = compute_marginal_probability(P, x0, n);
	
	print(x);

test_marginal_probability();
