__author__ = 'julie'
# -*- coding: utf-8 -*-
import O1_secretary_problem as O1
import O2_avalanche_problem as O2

import matplotlib.pyplot as plt
import numpy as np
import random as rnd


if __name__ == "__main__":
    Master_Flag = {
                    0: '1B) PlotBestCandidate_task 1b',
                    1: '1C) NumberOfCandidates_task 1c',
                    2: '1D) Plot 1 taskd1()',
                    3: '1D) Plot 2 taskd2(10,1000,3)',
                    4: '2A) test_marginal_probability()',
                    5: '2B) test_realizations()',
                    6: '2C) plot_forw_back_prob()',
                    7: '2D) test_cost()',
                    8: '2E) find_best_sensor_location()',


            }[6]
    if Master_Flag == '1B) PlotBestCandidate_task 1b':
        O1.plotBestCandidate(30)
        O1.plotBestCandidate(40)

        plt.show()

    elif Master_Flag == '1C) NumberOfCandidates_task 1c':
        n = 30
        k = int(n / np.e)
        realizations = 1000
        top_number = 3

        best_candidate1, top_number_candidate1, interview_all1 = O1.number_of_candidates_task1c(k, n, realizations,
                                                                                                top_number)
        best_candidate2, top_number_candidate2, interview_all2 = O1.number_of_candidates_task1c(k, n, realizations,
                                                                                                top_number)
        best_candidate3, top_number_candidate3, interview_all3 = O1.number_of_candidates_task1c(k, n, realizations,
                                                                                                top_number)

        print("The strategy gets the best candidate: ", best_candidate1 / realizations)
        print("The strategy give a candidate among the top " + str(top_number) + " values: ",
              top_number_candidate1 / realizations)
        print("The strategy end up interviewing all candidates: ", interview_all1 / realizations)
        print("The strategy gets the best candidate: ", best_candidate2 / realizations)
        print("The strategy give a candidate among the top " + str(top_number) + " values: ",
              top_number_candidate2 / realizations)
        print("The strategy end up interviewing all candidates: ", interview_all2 / realizations)
        print("The strategy gets the best candidate: ", best_candidate3 / realizations)
        print("The strategy give a candidate among the top " + str(top_number) + " values: ",
              top_number_candidate3 / realizations)
        print("The strategy end up interviewing all candidates: ", interview_all3 / realizations)


    elif Master_Flag == '1D) Plot 1 taskd1()':
        print("Optimal k-value for uniformly distributed nr. of candidates: ", O1.taskd1())
        plt.show()

    elif Master_Flag == '1D Plot 2 taskd2(10,1000,3)':
        O1.taskd2(10, 1000, 3)
        plt.show()

    elif Master_Flag == '2A) test_marginal_probability()':
        O2.test_marginal_probability()

    elif Master_Flag == '2B) test_realizations()':
        O2.test_realizations()

    elif Master_Flag == '2C) plot_forw_back_prob()':
        O2.plot_forw_back_prob()

    elif Master_Flag == '2D) test_cost()':
        O2.test_cost()

    elif Master_Flag == '2E) find_best_sensor_location()':
        print("Optimal sensor location is at section : ", O2.find_best_sensor_location())



