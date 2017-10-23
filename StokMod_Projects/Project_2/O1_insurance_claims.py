import matplotlib.pyplot as plt
import numpy as np
import math as math
import random as rnd



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
		N_t[t] = np.random.poisson(lam=lamb, size=1) # Here we draw a single sample from a Poisson distribu 
		N_t[t] += N_t[t-1]
	return N_t

def simulate_N_as_func_of_time_B():
	N_t = np.zeros(59)
	for t in range(0,59):
		lamb = 2 + np.cos((np.pi/182.5)*t)
		N_t[t] = np.random.poisson(lam=lamb, size=1)
		N_t[t] += N_t[t-1]
	return N_t

# vec = np.zeros((30, 3)) # 3 rader 30 kolonner

def plot_simulations(task, nrSimulations):
	b = int(nrSimulations)
	t = np.arange(0,59) 
	N = np.zeros((b,59))
	if (task == 1):
		for i in range(0,b):
			N[i,:] = simulate_N_as_func_of_time_A(3)
			plt.plot(t, N[i,:])
		plt.xlabel("t")
		plt.ylabel("N(t)")
		plt.title("Realizations of N(t) with constant intensity")
		plt.grid()
		return 1
	elif (task == 2):
		for i in range(0,b):
			N[i,:] = simulate_N_as_func_of_time_B()
			plt.plot(t, N[i,:])
		plt.xlabel("t")
		plt.ylabel("N(t)")
		plt.title("Realizations of N(t) with time dependent intensity")
		plt.grid()
		return 1
	else:
		print('What task are you talking about?')
		return 0

# plot_simulations(1,100)
# plt.show()

def simulateNrTasks2(nrSimulations, probLimit):
	n = nrSimulations # Number of simulations
	counter = 0
	length = 59
	for i in range(0,n):
		N_t = simulate_N_as_func_of_time_B()
		if (N_t[length-1] > probLimit):
			counter += 1
	percentage = (counter/n)*100
	print(percentage)


simulateNrTasks2(20000,175)
# print(simulate_N_as_fun100_of_time_B())
