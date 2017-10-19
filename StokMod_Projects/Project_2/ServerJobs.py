__author__ = 'julie'
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
#variables="example_page284"
variables="project_variables"

if (variables=="project_variables"):
    k_number_of_units=32
    arrival_rate_lambda=25
    exp_time_rate_mu=1
    rho=arrival_rate_lambda/(k_number_of_units*exp_time_rate_mu)

def compute_PI_ZERO(k,rho):
    sum=0
    for i in range(k+1):
        sum+=(k*rho)**i/math.factorial(i) #+   ((k**k)/math.factorial(k))*(rho**k)/(1-rho)
    return 1/sum

PI_ZERO=compute_PI_ZERO(k_number_of_units,rho)

def compute_eq_prob(k,l,m,n,PI_ZERO):
    return 1/math.factorial(n)*(l/m)**n*PI_ZERO

def plot_eq_probabilities(k,l,m,PI_ZERO):
    x=np.linspace(0,k,k+1)
    y=[0]*(k+1)
    integral_sum=0
    expected_value=0
    for i in range(len(y)):
        y[i]=compute_eq_prob(k,l,m,i,PI_ZERO)
        integral_sum+=y[i]
        expected_value+=y[i]*i
    print(y)
    print("exp_value",expected_value)
    print("integral sum",integral_sum)
    plt.plot(x,y)
    plt.title("Equilibrium Probabilities")
    plt.ylabel("P(N(t)=n)")
    plt.xlabel("n")
    plt.savefig("equilibrium_prob.pdf")
    plt.show()
print("pi_zero",PI_ZERO)
plot_eq_probabilities(k_number_of_units,arrival_rate_lambda,exp_time_rate_mu,PI_ZERO)


