__author__ = 'julie'
# -*- coding: utf-8 -*-
#import O1_secretary_problem as O1
import ServerJobs as O2

import numpy as np
import matplotlib.pyplot as plt
import math

k_number_of_units = 32
arrival_rate_lambda = 25
exp_time_rate_mu = 1
t_max=7*24
rho = arrival_rate_lambda / exp_time_rate_mu
PI_ZERO = O2.compute_PI_ZERO(k_number_of_units, rho)

realizations=100
if __name__ == "__main__":

    Master_Flag = {
                    0: '2A) Plot Equilibrium Probabilities',
                    1: '2B) Simulate N',
                    2: '2B) Plot Vector Time'



            }[1]        #<-------Write number of the function you want to test. For example, for finding the best sensor location, write 8 in the [ ].
    if Master_Flag =='2A) Plot Equilibrium Probabilities':
        O2.plot_eq_probabilities(k_number_of_units, arrival_rate_lambda, exp_time_rate_mu, PI_ZERO)

    elif Master_Flag=='2B) Simulate N':
        average_jobs_per_hour, average_T_vektor, t_list, average_n_list, iterations=O2.simulate_N(k_number_of_units, arrival_rate_lambda, exp_time_rate_mu, realizations,t_max)
        print("Average jobs per hour forwarded:",average_jobs_per_hour)
        plt.figure()
        O2.plot_transient(k_number_of_units,arrival_rate_lambda,exp_time_rate_mu,realizations,t_max)
        plt.show()

    elif Master_Flag=="2B) Plot Vector Time":

        plt.figure()
        O2.plot_vector_time(k_number_of_units,arrival_rate_lambda,exp_time_rate_mu,realizations,t_max)
        plt.show()



