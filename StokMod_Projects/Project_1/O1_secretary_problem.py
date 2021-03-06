__author__ = 'julie'
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import random as rnd


def plotBestCandidate(n, farge):
    k = np.linspace(1, n)
    func1 = k / n * np.log(n / k)

    plt.plot(k, func1, label=("n = "+ str(n)))
    print((int(n/np.e)) / n * np.log(n / int(n/np.e)))
    plt.xlabel("Candidate number k")
    plt.ylabel("Probability")
    plt.title("Probability of best candidate")
    plt.legend()
    plt.grid()


def secretary_problem_strategy(k, x_values):
    best_cand_of_k = max(x_values[:k])  # return the value of the best candidate up to k.
    for i in range(k,len(x_values)):
        if x_values[i] > best_cand_of_k:
            return i
    return len(x_values)-1              # last index.


def number_of_candidates_task1c(k, n, realizations,top_number):  # top_number = gives a candidate among the "top three" (top number)
    bestCandidateCounter = 0
    top_number_counter = 0
    interview_all_counter = 0
    top_number-=1                                                #reduces "top number" because we want to find the best and the next two best values. That's among the top 3 best values.
    for i in range(realizations):
        x_values = rnd.sample(range(1, n + 1), n)                # makes a list with n unique values between 1 and n.
        strategy_index = secretary_problem_strategy(k, x_values)

        if strategy_index == len(x_values)-1:
            interview_all_counter += 1

        if strategy_index == x_values.index(max(x_values)):
            bestCandidateCounter += 1
            top_number_counter += 1

        else:
            number = 0                                # Keeps track on how many times the while-loop has been looped.
            keep_on = True                            # This is true while the highest index is still not found.
            while number < top_number and keep_on:
                number += 1
                x_values[x_values.index(max(x_values))] = -1
                if strategy_index == x_values.index(max(x_values)):
                    top_number_counter += 1
                    keep_on = False                     # stop the for-loop if the index is found.


    return bestCandidateCounter, top_number_counter, interview_all_counter

def taskd1():
    k = np.linspace(1, 15, num=15)
    func1 = np.zeros(15)

    # The for-loop below sums up the probabilities of picking the best candidate for the individual k-values
    for n in range(16, 46):
        func1 = func1 + k / (30 * n) * np.log(n / k)

    # Here P(Z=1) is plotted as a function of k
    plt.plot(k, func1, label=("P(Z=1)"))
    plt.xlabel("Candidate number k")
    plt.ylabel("Probability")
    plt.title("Probability of picking the best candidate")
    plt.legend()
    plt.grid()
    k_max = max(func1)
    for i in range(15):
        if func1[i] == k_max:
            return k[i]


def taskd2(k, realizations, top_number):
    top_number-=1                #reduces "top number" because we want to find the best and the next two best values. That's among the top 3 best values.
    vec = np.zeros((30, 3))
    for i in range(realizations):
        n = rnd.randint(16, 45) # Here we generate a random number of candidates (n)
        x_values = rnd.sample(range(1, n + 1), n)  # makes a list with n unique values between 1 and n.
        strategy_index = secretary_problem_strategy(k, x_values)

        # The if-statements below decide whether the secretary_problem_strategy() function chose the best candidate,
        # one of the top three candidates or the last candidate
        if strategy_index == len(x_values)-1:
            vec[n - 16, 0] += 1


        elif strategy_index == x_values.index(max(x_values)):
            vec[n - 16, 1] += 1
            vec[n - 16, 2] += 1

        else:
            number = 0
            keep_on = True
            while (number < top_number and keep_on):
                number += 1
                x_values[x_values.index(max(x_values))] = -1
                if strategy_index == x_values.index(max(x_values)):
                    vec[n - 16, 2] += 1
                    keep_on = False
                # stop the for-loop if the index is found.

    # Here we plot the different bar graphs
    x = range(16, 46)
    width = 1 / 1.5
    fig1 = plt.bar(x, vec[:, 0], width, color="red", label=("Last candidate"))
    plt.xlabel("Realizations of n")
    plt.ylabel("# picks")
    plt.title("Times the last candidate was picked")
    plt.legend()
    plt.grid()
    plt.show()
    fig2 = plt.bar(x, vec[:, 1], width, color="green", label=("Best candidate"))
    plt.xlabel("Realizations of n")
    plt.ylabel("# picks")
    plt.title("Times the best candidate was picked")
    plt.legend()
    plt.grid()
    plt.show()
    fig3 = plt.bar(x, vec[:, 2], width, color="blue", label=("Top three candidates"))
    plt.xlabel("Realizations of n")
    plt.ylabel("# picks")
    plt.title("Times the top " + str(top_number) + " best candidates were picked")
    plt.legend()
    plt.grid()
    nrBestCandidates = 0
    nrTopCandidates = 0
    nrFailures = 0
    for i in range(30):
        nrFailures += vec[i, 0]
        nrBestCandidates += vec[i, 1]
        nrTopCandidates += vec[i, 2]
    return nrBestCandidates, nrTopCandidates, nrFailures

