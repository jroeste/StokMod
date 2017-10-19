import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rnd


def poissonProb(l,t,min,max):
	if min == max:
		return math.exp(-l*t)*((l*t)^min)/math.factorial(min)
	prob = 0
	for i in range(min,max):
		temp = math.exp(-l*t)*((l*t)^i)
		for j in range(1,i):
			temp /= j
		prob += temp
	return prob

print(poissonProb(3,59,0,1)


