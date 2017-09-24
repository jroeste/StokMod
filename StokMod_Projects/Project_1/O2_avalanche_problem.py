__author__ = 'julie'
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.style as sty
import numpy as np
import numpy.random as rnd

def compute_marginal_probability(P, x0, n):
	dim = len(x0);
	x = np.zeros((n, dim));
	
	x[0,:] = x0;
	for i in range(0, n - 1):
		x[i + 1,:] = np.matmul(P, x[i,:]);
	return x;

def sample(p):
	s = rnd.rand();
	p_sample = [s, 1 - s];
	return np.less(p_sample, p).astype(np.float64);

def draw_realization(P, x0, n):
	dim = len(x0);
	r = np.zeros((n, dim));

	# Get initial state from initial probability distribution
	r[0,:] = sample(x0);
	for i in range(0, n - 1):
		# Get probability distribution for next state
		p = np.matmul(P, r[i,:]);
		r[i + 1,:] = sample(p);
	return r;

def draw_realizations(P, x0, n, r):
	dim = len(x0);
	x = np.zeros((r, n, dim));

	for i in range(1, r):
		x[i,:,:] = draw_realization(P, x0, n);
	return x;

def test_marginal_probability():
	# Define left stochastic matrix
	P = [[0.95, 0.00], [0.05, 1]];
	x0 = [0.99, 0.01];

	n = 50;
	x = compute_marginal_probability(P, x0, n);

	fig = plt.figure();
	ax = fig.gca();

	ax.plot(range(0,n), x[:,0], color="black");

	ax.set_title("Calculation of Low Risk Probability versus Section");
	ax.set_xlabel("Section");
	ax.set_ylabel("Probability");
	ax.minorticks_on();
	ax.set_xticks([0, n - 1]);
	ax.set_yticks([0, 1]);
	ax.set_xlim([0, n - 1]);
	ax.set_ylim([0, 1]);
	ax.set_axisbelow(True);
	ax.grid(which="both");

	fig.savefig("calculated.pdf");
	fig.show();

def test_realizations():
	# Define left stochastic matrix
	P = [[0.95, 0.00], [0.05, 1]];
	x0 = [0.99, 0.01];

	n = 50;
	r = 25;

	x = compute_marginal_probability(P, x0, n);
	z = draw_realizations(P, x0, n, r);
	p = np.sum(z[:,:,1], axis=0) / r;	

	fig = plt.figure();
	ax = fig.gca();

	ax.matshow(z[:,:,0], origin="bottom", cmap="hot");

	ax.set_title("Indepenent realization of risk state versus section");
	ax.set_xlabel("Section");
	ax.set_ylabel("Realization");
	ax.minorticks_on();
	ax.set_xticks([0, n - 1]);
	ax.set_yticks([0, r - 1]);
	ax.set_xticks([x - 0.5 for x in range(1, n)], minor="true");
	ax.set_yticks([y - 0.5 for y in range(1, r)], minor="true"); 
	ax.xaxis.set_ticks_position("bottom");
	ax.set_axisbelow(True);
	ax.grid(which="minor");

	fig.savefig("state.pdf");
	fig.show();

	fig = plt.figure();
	ax = fig.gca();

	b = ax.bar(range(n), p, color="black", label="Sampled mean");
	l, = ax.plot(range(0,n), x[:,1], color="grey", label="Calculated probability");
	ax.legend(loc=4);

	ax.set_title("Probability of High Risk versus Section");
	ax.set_xlabel("Section");
	ax.set_ylabel("Probability");
	ax.minorticks_on();
	ax.set_xticks([0, n]);
	ax.set_yticks([0, 1]);
	ax.set_xlim([0, n]);
	ax.set_ylim([0, 1]);
	ax.set_axisbelow(True);
	ax.grid(which="both");

	fig.savefig("sampled.pdf");
	fig.show();

def test_cost():
	P = [[0.95, 0.00], [0.05, 1]];
	x0 = [0.99, 0.01];

	n = 50;
	x = compute_marginal_probability(P, x0, n);
	
	repair_cost = 5000;

	individual_cost = repair_cost*sum(x[0,:]);
	all_cost = 100000;
	cost = min(individual_cost, all_cost);
	print(individual_cost);
	print(all_cost);

#sty.use("seaborn-poster");
test_marginal_probability();
test_realizations();
test_cost();
