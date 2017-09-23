__author__ = 'julie'
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

# Oppgave 2a)
def compute_marginal_probability(P, x0, n):
    dim = len(x0);

    x = np.zeros((dim, n));

    x[:, 0] = x0;
    for i in range(0, n - 1):
        x[:, i + 1] = np.matmul(P, x[:, i]);
    return x;


# Oppgave 2b)
def sample(p):
    s = rnd.rand();
    p_sample = [s, 1 - s];
    return np.less(p_sample, p).astype(np.float64);


def draw_realization(P, x0, n):
    dim = len(x0);
    r = np.zeros((dim, n));

    r[:, 0] = sample(x0);
    for i in range(0, n - 1):
        p = np.matmul(P, r[:, i]);
        r[:, i + 1] = sample(p);
    return r;


def draw_realizations(P, x0, n, r):
    dim = len(x0);
    x = np.zeros((r, n));

    for i in range(1, r):
        x[i, :] = draw_realization(P, x0, n)[0];
    return x;


# Oppgave 2c) del 1
def compute_forward__probability(P, x0, initial_cond, n, k):
    dim = len(initial_cond);
    x_l = np.zeros((dim, n));
    x_l[:, k] = initial_cond
    for i in range(k, n - 1):
        x_l[0, i + 1] = 1 - (P[0][0]) * (1 - x_l[0][i])  # p(x_l=2|x_k=1)=1-0.95*(1-P(x_(l-1)
        x_l[1, i + 1] = 1 - (P[0][1]) * (1 - x_l[1][i])

    return x_l;


# Oppgave 2c) del 2
def compute_backward_probability(P, x0, initial_cond, n, k):
    dim = len(initial_cond);
    x_l = np.zeros((dim, n));
    x = compute_marginal_probability(P, x0, n)
    x_l[:, k - 1] = initial_cond  # initial for number k
    for i in range(k - 1, -1, -1):
        x_l[0, i] = P[0][1] * x[0, i] / x[0, k - 1]  # blir 0
        x_l[1, i] = P[1][1] * x[1, i] / x[1, k - 1];

    return x_l;


def plot_forw_back_prob():
    P = [[0.95, 0.00], [0.05, 1]];  # Transition Matrix
    x0 = [0.99, 0.01];  # Initial probability on low and high risk
    initial_cond = [0, 1]  # Initial condition for sensor location
    k = 20  # Sensor location
    n = 50

    x_f = compute_forward__probability(P, x0, initial_cond, n, k)
    x_b = compute_backward_probability(P, x0, initial_cond, n, k)

    plt.plot(range(k, n), x_f[0, k:n], 'o-', label='Forward Pr.(i=1)')
    plt.plot(range(k, n), x_f[1, k:n], 'o-', label='Forward Pr.(i=2)')
    plt.plot(range(1, k + 1), x_b[0, :k], 'o-', label='Backward Pr.(i=1)')
    plt.plot(range(1, k + 1), x_b[1, :k], 'o-', label='Backward Pr.(i=2)')

    plt.ylim((-0.1, 1.1))
    plt.xlabel("Number $l$")
    plt.ylabel("Probability")
    plt.title("Probability with sensor at location k")
    plt.legend(loc=4)
    plt.show()



def test_marginal_probability():
    P = [[0.95, 0.00], [0.05, 1]];
    x0 = [0.99, 0.01];

    n = 50;
    x = compute_marginal_probability(P, x0, n);


# plt.plot(range(0,n), x[1,:]);
# plt.savefig("teest.png");
# print(x);

def test_realizations():
    P = [[0.95, 0.00], [0.05, 1]];
    x0 = [0.99, 0.01];

    n = 50;
    r = 25;

    x = draw_realizations(P, x0, n, r);
    plt.matshow(x, origin="bottom", cmap="hot");
    #plt.gca().XAxis.tick_bottom();
    #plt.gca().YAxis.tick_left();



    plt.gca().minorticks_on();
    plt.gca().set_xticks([0, n - 1]);
    plt.gca().set_yticks([0, r - 1]);
    plt.gca().set_xticks([x - 0.5 for x in range(1, n)], minor="true");
    plt.gca().set_yticks([y - 0.5 for y in range(1, r)], minor="true");
    plt.grid(which="minor");
    plt.show();


def test_cost():
    P = [[0.95, 0.00], [0.05, 1]];
    x0 = [0.99, 0.01];

    n = 50;
    x = compute_marginal_probability(P, x0, n);
    print(x)

    repair_cost = 5000;

    individual_cost = repair_cost * sum(x[1, :]);
    all_cost = 100000;
    cost = min(individual_cost, all_cost);
    print(individual_cost);
    print(all_cost);

def find_best_sensor_location():
    P = [[0.95, 0.00], [0.05, 1]];
    x0 = [0.99, 0.01];

    n = 50;
    x = compute_marginal_probability(P, x0, n);
    initial_cond = [0, 1]

    repair_cost = 5000;
    individual_cost = repair_cost * sum(x[1, :]);
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
                    individual_cost_of_k+=compute_backward_probability(P,x0,initial_cond,n,k)[i,j]
                elif j>=k:
                    individual_cost_of_k+=compute_forward__probability(P,x0,initial_cond,n,k)[i,j]

            V_k+=min(all_cost,individual_cost_of_k*repair_cost)*x[i,k]
        V[k-1]=V_k
    print(V)
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
    plt.legend()
    plt.show()

    return sensor_location

#plot_forw_back_prob()

#print(find_best_sensor_location())

#test_marginal_probability();
#test_realizations();
#test_cost();