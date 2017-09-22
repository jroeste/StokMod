__author__ = 'julie'
# -*- coding: utf-8 -*-
import O1_secretary_problem as O1
import O2_avalanche_problem as O2

import matplotlib.pyplot as plt
import numpy as np
import random as rnd


if __name__ == "__main__":
    Master_Flag = {
                    0: 'PlotBestCandidate_task 1b',
                    1: 'NumberOfCandidates_task 1c',
                    2: 'Task 1d'
            }[0]
    if Master_Flag == 'PlotBestCandidate_task 1b':
        O1.plotBestCandidate(30)
        O1.plotBestCandidate(40)

        plt.show()

    elif Master_Flag == 'NumberOfCandidates_task 1c':
        n=30
        k=int(n/np.e)
        realizations=1000
        top_number=3
        best_candidate,top_number_candidate,interview_all=O1.number_of_candidates_task1c(k, n, realizations, top_number)
        print("The strategy gets the best candidate: ", best_candidate/realizations)
        print("The strategy give a candidate among the top three values: ", top_number_candidate/realizations)
        print("The strategy end up interviewing all candidates: ", interview_all/realizations)

    elif Master_Flag == 'Task 1d':
        print("test")

