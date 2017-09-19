__author__ = 'julie'
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random as rnd


def plotBestCandidate(n):
	k = np.linspace(1, n)
	func1 = k / n * np.log(n / k)

	plt.plot(k, func1, label=("n=", n))
	plt.xlabel("Candidate number k")
	plt.ylabel("Probability")
	plt.title("Probability of best candidate")
	plt.legend()
	plt.grid()


def secretary_problem_strategy(k, x_values):
    best_cand_of_k = max(x_values[:k])  # return the value of the best candidate up to k.
    for i in range(len(x_values)):
        if x_values[i] > best_cand_of_k:
            return i
    return False  # last index.


def number_of_candidates_task1c(k, n, realizations, top_number):  # top_number = gives a candidate among the "top three" (top number)
    bestCandidateCounter = 0
    top_three_counter = 0
    interview_all_counter = 0
    for i in range(realizations):
        x_values = rnd.sample(range(1, n + 1), n)  # makes a list with n unique values between 1 and n.
        strategy_index = secretary_problem_strategy(k, x_values)

        if strategy_index == False:
            interview_all_counter += 1


        elif strategy_index == x_values.index(max(x_values)):
            bestCandidateCounter += 1
            top_three_counter += 1

        else:

            number = 0
            keep_on = True
            while (number < top_number and keep_on):  # test this with a short list.
                number += 1
                x_values[x_values.index(max(x_values))] = -1

                if strategy_index == x_values.index(max(x_values)):
                    top_three_counter += 1

                    keep_on = False
                    # stop the for-loop if the index is found.

    return bestCandidateCounter, top_three_counter, interview_all_counter

def plotZ1_of_k_unknown_n():
    print("hei")


def func(k,n): 
	return k / n * np.log(n / k)

def taskd():
	k = np.linspace(1, 15, num=15)
	func1 = np.zeros(15)
	for n in range(16,30):
		func1 = func1 + k / n * np.log(n / k)
	plt.plot(k, func1, label=("P(Z=1)", n))
	plt.xlabel("Candidate number k")
	plt.ylabel("Probability")
	plt.title("Probability of best candidate")
	plt.legend()
	plt.grid()
	k_max = max(func1)
	for i in range(15):
		if func1[i] == k_max:
			return i

print(taskd())
plt.show()