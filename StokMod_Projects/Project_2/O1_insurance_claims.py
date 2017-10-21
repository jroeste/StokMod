import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rnd


# def poissonProb(l,t,min,max):
# 	if min == max:
# 		return math.exp(-l*t)*((l*t)^min)/math.factorial(min)
# 	prob = 0
# 	for i in range(min,max):
# 		temp = math.exp(-l*t)*((l*t)^i)
# 		for j in range(1,i):
# 			temp /= j
# 		prob += temp
# 	return prob

# print(poissonProb(3,59,0,1)
def simulateNrTasks(nrSimulations, probLimit):
	n = nrSimulations # Number of simulations
	lam = 3 # Intensity constant
	t = 59 # Time interval we are interested inn (59 days)
	N = np.random.poisson(lam=lam*t, size=n) #Sample the number of insurance claims from the poisson distriution.
	counter = 0
	for i in range(0,n):
			if (N[i] > probLimit):
				counter += 1
	percentage = (counter/n)*100
	print(percentage, '%')

simulateNrTasks(10000,175)
