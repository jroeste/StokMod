__author__ = 'julie'
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


#import F_xxxxxxxxx
#import G_xxxxxxxxx

import H_BrownianMotion as H

mean=0
variance=0.75
start_time=0
end_time=120
initial_stock_price=40
wished_stock_price=50
realizations=100
time_limit=10000


if __name__ == "__main__":

    Master_Flag = {
                    0: 'H) Stock Price probability 1',
                    1: "H) Stock Price probability 2",
                    2: 'H) Hitting Times',


            }[1]        #<-------Write number of the function you want to test. For example, for finding the best sensor location, write 8 in the [ ].
    if Master_Flag =='H) Stock Price probability 1':
        start_time1=0   #Jan 1st

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

    elif Master_Flag=='Hitting Times':
        H.waiting_time(mean, variance, 0.02, initial_stock_price, start_time, realizations, time_limit)
        plt.show()

