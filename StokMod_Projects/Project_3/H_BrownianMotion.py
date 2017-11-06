__author__ = 'julie'
# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def stock_price_probability(mean,variance,start_time,end_time, initial_stock_price,wished_stock_price,realizations):
    days=end_time-start_time
    new_wished_stock_price=wished_stock_price-initial_stock_price
    number_of_stock_price_over_wished=0
    for i in range (realizations):
        t = np.linspace(start_time, end_time, days+1)
        x = np.zeros(days+1)
        for j in range(1,days+1):
            z_i=np.random.normal()
            x[j]=x[j-1]+z_i*variance

        if x[-1]>=new_wished_stock_price:
            number_of_stock_price_over_wished+=1
        plt.plot(t,x+initial_stock_price)
    plt.xlabel("Days")
    plt.ylabel("Stock Price")

    return number_of_stock_price_over_wished/realizations




def waiting_time(mean,variance,rate_of_change,initial_stock_price,start_time,realizations,time_limit):
    wished_stock_price=(1+rate_of_change)*initial_stock_price
    hitting_times = np.zeros(realizations)
    x=np.zeros(time_limit+1)
    t_list=np.linspace(start_time,realizations,realizations)
    x[0]=initial_stock_price

    #Plot cumulative distrubution function:
    cumulative=np.zeros(realizations+1)
    for i in range(1,realizations+1):
        cumulative[i]=2*(1-norm.cdf((rate_of_change*initial_stock_price)/np.sqrt(time_limit/realizations*i*variance**2)))



    #Plott hitting time function
    for i in range(realizations):
        time=0
        while x[time]<wished_stock_price and time<10000:
            time += 1
            x[time]= x[time-1]+np.random.normal()*variance  #stock price
        hitting_times[i]=time

    mean_number=np.mean(hitting_times)
    print("Average - mean number: ",mean_number)
    std_number=np.std(hitting_times)
    print("Standard Deviation: ",std_number)
    hitting_times.sort()

    x_list = np.linspace(0, len(cumulative), len(cumulative))
    plt.plot(x_list * (time_limit / realizations), cumulative,label="Analytic - cdf")
    plt.plot(hitting_times,t_list/realizations,label="Simulated - Hitting Times")
    plt.ylabel("Probability")
    plt.xlabel("Waiting time")
    plt.legend(loc=4)














