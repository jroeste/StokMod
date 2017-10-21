__author__ = 'julie'
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import matplotlib.style as sty
import numpy as np
import numpy.random as rnd


def compute_marginal_probability(P, x0, n):
    dim = len(x0);
    # Create table for P(X_i = j) at x[i,j]
    x = np.zeros((n, dim));

    x[0, :] = x0;
    for i in range(0, n - 1):
        x[i + 1, :] = np.matmul(P, x[i, :]);

    return x;


def sample(p):
    # Take probability distribution p and a vector with 1 in the active state
    s = rnd.rand();
    p_sample = [s, 1 - s];
    return np.less(p_sample, p).astype(np.float64);


def draw_realization(P, x0, n):
    dim = len(x0);
    r = np.zeros((n, dim));

    # Get initial state from initial probability distribution
    r[0, :] = sample(x0);
    for i in range(0, n - 1):
        # Get probability distribution for next state
        p = np.matmul(P, r[i, :]);
        r[i + 1, :] = sample(p);
    return r;


def draw_realizations(P, x0, n, r):
    dim = len(x0);
    x = np.zeros((r, n, dim));

    for i in range(1, r):
        x[i, :, :] = draw_realization(P, x0, n);
    return x;

# Oppgave 2c) del 1
def compute_forward__probability(P, x0, initial_cond, n, k):
    dim = len(initial_cond);
    x_l = np.zeros((n, dim));
    x_l[k,:] = initial_cond

    for i in range(k, n - 1):
        #x_l[i + 1,0] = 1 - (P[0][0]) * (1 - x_l[i,0])  # p(x_l=2|x_k=1)=1-0.95*(1-P(x_(l-1)
        x_l[i+1,0]=P[1][0]+P[0][0]*x_l[i,0]
        #x_l[i + 1,1] = 1 - (P[0][1]) * (1 - x_l[i,1])
        x_l[i + 1, 1] = P[1][0] + P[0][0] * x_l[i, 1]

    return x_l;


# Oppgave 2c) del 2
def compute_backward_probability(P, x0, initial_cond, n, k):
    dim = len(initial_cond);
    x_l = np.zeros((n, dim));
    x = compute_marginal_probability(P, x0, n)
    x_l[k - 1,:] = initial_cond  # initial for number k
    for i in range(k - 1, -1, -1):
        x_l[i,0] = P[0][1] * x[i,0] / x[k - 1,0]  # blir 0
        x_l[i,1] = P[1][1] * x[i,1] / x[k - 1,1];

    return x_l;


def plot_forw_back_prob():
    P = [[0.95, 0.00], [0.05, 1]];  # Transition Matrix
    x0 = [0.99, 0.01];  # Initial probability on low and high risk
    initial_cond = [0, 1]  # Initial condition for sensor location
    k = 20  # Sensor location
    n = 50

    x_f = compute_forward__probability(P, x0, initial_cond, n, k)
    x_b = compute_backward_probability(P, x0, initial_cond, n, k)

    plt.plot(range(k, n), x_f[k:n,0], 'o-', label='Forward Pr.(i=1)')
    plt.plot(range(k, n), x_f[k:n,1], 'o-', label='Forward Pr.(i=2)')
    plt.plot(range(1, k + 1), x_b[:k,0], 'o-', label='Backward Pr.(i=1)')
    plt.plot(range(1, k + 1), x_b[:k,1], 'o-', label='Backward Pr.(i=2)')

    plt.ylim((-0.1, 1.1))
    plt.xlabel("Number $l$")
    plt.ylabel("Probability")
    plt.title("Probability with sensor at location k")
    plt.legend(loc=4)
    plt.savefig("forw_back_prob_2c", filetype=".png")
    plt.show()

def test_marginal_probability():
    # Define left stochastic matrix
    P = [[0.95, 0.00], [0.05, 1]];
    x0 = [0.99, 0.01];

    n = 50;
    x = compute_marginal_probability(P, x0, n);

    fig = plt.figure();
    ax = fig.gca();

    ax.plot(range(0, n), x[:, 0], color="black");

    ax.set_title("Calculation of High Risk Probability versus Section");
    ax.set_xlabel("Section");
    ax.set_ylabel("Probability");
    ax.minorticks_on();
    ax.set_xticks([0, n - 1]);
    ax.set_yticks([0, 1]);
    ax.set_xlim([0, n - 1]);
    ax.set_ylim([0, 1]);
    ax.set_axisbelow(True);
    ax.grid(which="both");

    #fig.savefig("calculated.pdf");
    fig.savefig("calculated",fileformat=".pdf")
    fig.show();


def test_realizations():
    # Define left stochastic matrix
    P = [[0.95, 0.00], [0.05, 1]];
    x0 = [0.99, 0.01];

    n = 50;
    r = 25;

    x = compute_marginal_probability(P, x0, n);
    z = draw_realizations(P, x0, n, r);
    p = np.sum(z[:, :, 1], axis=0) / r;

    fig = plt.figure();
    ax = fig.gca();

    ax.matshow(z[:, :, 0], origin="bottom", cmap="hot");

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

    #fig.savefig("state.pdf");
    fig.savefig("state",fileformat=".pdf")
    fig.show();

    fig = plt.figure();
    ax = fig.gca();

    b = ax.bar(range(n), p, color="black", label="Sampled mean");
    l, = ax.plot(range(0, n), x[:, 1], color="grey", label="Calculated probability");
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

    #fig.savefig("sampled.pdf");
    fig.savefig("sampled",fileformat=".pdf")
    fig.show();


def test_cost():
    # Define left stochastic matrix
    P = [[0.95, 0.00], [0.05, 1]];
    x0 = [0.99, 0.01];

    n = 50;
    x = compute_marginal_probability(P, x0, n);

    repair_cost = 5000;

    individual_cost = repair_cost * sum(x[:, 1]);
    all_cost = 100000;
    cost = min(individual_cost, all_cost);
    print("Individual Cost:",individual_cost);
    print("All Cost",all_cost);

def find_best_sensor_location():
    P = [[0.95, 0.00], [0.05, 1]];
    x0 = [0.99, 0.01];

    n = 50;
    x = compute_marginal_probability(P, x0, n);
    initial_cond = [0, 1]

    repair_cost = 5000;
    individual_cost = repair_cost * sum(x[:,1]);
    all_cost = 100000;

    V=np.zeros(n)
    individual_cost_of_k=0
    k_list=np.linspace(1,n+1)

    for k in range(0,n):
        V_k = 0
        individual_cost_of_k=0
        for i in range(2):
            for j in range(0, n):
                if j<k:
                    individual_cost_of_k+=compute_backward_probability(P,x0,initial_cond,n,k)[j,i]
                elif j>=k:
                    individual_cost_of_k+=compute_forward__probability(P,x0,initial_cond,n,k)[j,i]

            V_k+=min(all_cost,individual_cost_of_k*repair_cost)*x[k,i]
        V[k-1]=V_k

    min_value = min(V)
    for i in range(len(V)):
        if V[i] == min_value:
            sensor_location=i+1

    #plotter:
    plt.plot(k_list[:-1],V[:-1],"o-",label="$V_k(k)$")
    plt.xlabel("k-values")
    plt.ylabel("Costs")
    plt.title("Expected gain information, $V_k$")
    plt.axvline(sensor_location,label="Optimal Sensor Location $k=30$",color="red")
    plt.savefig("Sensor_location_2e",fileformat=".pdf")
    plt.legend()
    plt.show()

    return sensor_location
# sty.use("seaborn-poster");
#test_marginal_probability();
#test_realizations();
#test_cost();
