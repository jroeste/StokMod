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
def simulateNrTasks(nrSimulations, probLimit, timePeriod, intensity):
	n = nrSimulations # Number of simulations
	lam = intensity # Intensity constant
	t = timePeriod # Time interval we are interested inn (59 days)
	N = np.random.poisson(lam=lam*t, size=n) #Sample the number of insurance claims from the poisson distriution.
	counter = 0
	for i in range(0,n):
			if (N[i] > probLimit):
				counter += 1
	percentage = (counter/n)*100
	return percentage

# print(simulateNrTasks(10000,175,59,3))

def simulate_N_as_func_of_time_A(lamb):
	N_t = np.zeros(59)
	for t in range(0,59):
		N_t[t] = np.random.poisson(lam=lamb*t, size=1)
	return N_t

def simulate_N_as_func_of_time_B():
	N_t = np.zeros(59)
	for t in range(0,59):
		lamb = 2 + np.cos((np.pi/182.5)*t)
		N_t[t] = np.random.poisson(lam=lamb*t, size=1)
	return N_t

# vec = np.zeros((30, 3)) # 3 rader 30 kolonner

def plot_simulations(task):
	b = 100
	t = range(0,59)
	N = np.zeros((100,59))
	if (task == 1):
		for i in range(0,b):
			N[i,:] = simulate_N_as_func_of_time_A(3)
			plt.plot(t, N[i,:], color='blue')
		plt.xlabel("t")
		plt.ylabel("N(t)")
		plt.title("Probability")
		plt.grid()
		return 1
	elif (task == 2):
		for i in range(0,b):
			N[i,:] = simulate_N_as_func_of_time_B()
			plt.plot(t, N[i,:])
		plt.xlabel("t")
		plt.ylabel("N(t)")
		plt.title("Probability")
		plt.grid()
		return 1
	else:
		print('What task are you talking about?')
		return 0

plot_simulations(2)
plt.show()
