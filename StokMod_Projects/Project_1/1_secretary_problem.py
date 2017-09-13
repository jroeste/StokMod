__author__ = 'julie'
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random as rnd

def plotBestCandidate(n):
    k=np.linspace(1,n)

    func1=k/n*np.log(n/k)

    plt.plot(k, func1,label=("n=",n))

    #plt.plot(k,dFunc1)
    plt.plot(k,0*k)
    plt.xlabel("Candidate number k")
    plt.ylabel("Probability")
    plt.title("Probability of best candidate")
    plt.legend()
    plt.grid()

plotBestCandidate(30)
plt.show()

def taskc(n):
    i=np.linspace(0,n-1)
    x_values=[]
    b=rnd.random.uniform(1,1000)
    for i in range (n):
        b = rnd.randint(1, 1000)
        x_values.append(b)




