__author__ = 'julie'
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



import TaskF as F
import TaskG as G
import H_BrownianMotion as H

mean=0
variance=0.75
start_time=0
end_time=120
initial_stock_price=40
wished_stock_price=50
realizations=1000
time_limit=10000


if __name__ == "__main__":

	Master_Flag = {
		0: 'F) Unconditional Gaussian Process',
		1: 'F) Conditional Gaussian Process',
		2: 'G) Expected Quality',
		3: 'G) Quality > 57',
		4: 'G) Introducing Additional Data Point',
		5: 'H) Stock Price probability 1',
		6: 'H) Stock Price probability 2',
		7: 'H) Hitting Times',
			}[2]		#<-------Write number of the function you want to test. For example, for finding the best sensor location, write 8 in the [ ].
	if Master_Flag =='H) Stock Price probability 1':
		start_time1=0	#Jan 1st

		plt.figure()
		plt.title("Stock Prices from Jan 1st to May 1st")
		print("P(x(120)>50|x(0)=40)", H.stock_price_probability(mean, variance, start_time1, end_time, 40, 50, realizations))
		plt.savefig("stock_price_jan_may.pdf")
		plt.show()



	elif Master_Flag=="H) Stock Price probability 2":
		start_time2 = 60  # March 2nd
		plt.figure()
		plt.title("Stock Prices from Mar 2nd to May 1st")
		print("P(x(120)>50|x(60)=45)",
			  H.stock_price_probability(mean, variance, start_time2, end_time, 45, 50, realizations))
		plt.savefig("stock_price_march_may.pdf")
		plt.show()

	elif Master_Flag=='H) Hitting Times':
		plt.title("Hitting Times, 10% increase from x(0)=40")
		H.waiting_time(mean, variance, 0.1, initial_stock_price, start_time, realizations, time_limit)
		plt.savefig("HittingTimes.pdfHittingTimes.pdf")
		plt.show()
	elif Master_Flag=='F) Unconditional Gaussian Process':
		F.F1();
	elif Master_Flag=='F) Conditional Gaussian Process':
		F.F2();
	elif Master_Flag=='G) Expected Quality':
		G.G1();
	elif Master_Flag=='G) Quality > 57':
		G.G2();
	elif Master_Flag=='G) Introducing Additional Data Point':
		G.G3();

