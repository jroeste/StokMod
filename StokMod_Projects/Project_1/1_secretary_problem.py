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


# plotBestCandidate(30)
# plt.show()

def secretary_problem_strategy(k, x_values):
    best_cand_of_k = max(x_values[:k])  # return the value of the best candidate up to k.
    for i in range(len(x_values)):
        if x_values[i] > best_cand_of_k:
            return i
    return False  # last index.


def taskc(k, n, realizations, top_number):  # top_number = gives a candidate among the "top three" (top number)
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


print(taskc(25, 30, 1000, 3))

sum1 = 0
sum2 = 0
sum3 = 0
for i in range(2):
    a, b, c = taskc(int(1000 / np.e), 1000, 1000, 3)
    sum1 += a
    sum2 += b
    sum3 += c
print(sum1 / 20)
print(sum2 / 20)
print(sum3 / 20)

# (3661, 5946, 3828)
