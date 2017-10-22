__author__ = 'julie'
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math



def compute_PI_ZERO(k,rho):
    sum=0
    for i in range(k+1):
        sum+=(k*rho)**i/math.factorial(i)
    return 1/sum


def compute_eq_prob(k,l,m,n,PI_ZERO):
    return 1/math.factorial(n)*(l/m)**n*PI_ZERO

def plot_eq_probabilities(k,l,m,PI_ZERO):
    x=np.linspace(0,k,k+1)
    y=[0]*(k+1)
    integral_sum=0
    expected_value=0
    for i in range(len(y)):
        y[i]=compute_eq_prob(k,l,m,i,PI_ZERO)
        integral_sum+=y[i]
        expected_value+=y[i]*i
    print("exp_value",expected_value)
    print("integral sum",integral_sum)
    plt.plot(x,y)
    plt.title("Equilibrium Probabilities")
    plt.ylabel("P(N(t)=n)")
    plt.xlabel("n")
    plt.savefig("equilibrium_prob.pdf")
    plt.show()


def simulate_N(k,l,m,realizations,t_max):
    average_n_list = np.zeros(1000000)
    average_n_list[0] = 0
    average_T_vektor=np.zeros(k+1)
    average_T_vektor[0]=0
    average_lost_jobs=0

    for i in range(realizations):
        n_list=np.zeros(100000)
        t_list=np.zeros(100000)
        T_vektor=np.zeros(k+1)
        state=0
        n_list[0]=0
        t_list[0]=0
        T_vektor[0]=0
        i = 0
        lost_jobs = 0

        while (t_list[i] < t_max):

            if n_list[i]==0:
                time_birth = np.random.exponential(1/l, 1)
                n_list[i+1]=n_list[i]+1
                t_list[i + 1] = t_list[i] + time_birth
                T_vektor[state] += time_birth
                average_T_vektor[state]+=time_birth

                average_n_list[i+1]+=n_list[i+1]
                state += 1

            elif (0<n_list[i]) and n_list[i]<k:

                time_birth = np.random.exponential(1/l, 1)
                time_death = np.random.exponential(1 / (n_list[i]*m), 1)
                time = min(time_birth, time_death)
                t_list[i + 1] = t_list[i] + time

                T_vektor[state] += time
                average_T_vektor[state] += time

                if time==time_birth:
                    n_list[i + 1] = n_list[i] + 1
                    state+=1
                else:
                    n_list[i + 1] = n_list[i] - 1
                    state-=1

                average_n_list[i + 1] += n_list[i + 1]


            elif n_list[i]==k:

                time_death = np.random.exponential(1 / (k*m), 1)
                time_birth= np.random.exponential(1/l, 1)
                time = min(time_birth, time_death)

                t_list[i + 1] = t_list[i] + time

                T_vektor[state] += time
                average_T_vektor[state] += time


                if time == time_birth:

                    lost_jobs += 1
                    average_lost_jobs += 1
                    n_list[i + 1] += n_list[i]              #not increasing n
                    average_n_list[i + 1] += n_list[i + 1]

                elif time==time_death:

                    n_list[i+1]=n_list[i]-1
                    state-=1
                    average_n_list[i + 1] += n_list[i + 1]
            i+=1


    average_lost_jobs/=realizations
    average_lost_jobs_per_hour=average_lost_jobs/t_max
    average_T_vektor/=realizations
    average_n_list /= realizations

    return average_lost_jobs_per_hour,average_T_vektor,t_list,average_n_list,i

def plot_transient(k,l,m,realizations,t_max):
    lb,average_T_vektor,t_list,average_n_list,iterations=simulate_N(k,l,m,realizations,t_max)
    plt.plot(t_list[:iterations],average_n_list[:iterations])
    plt.title("System reaching equilibrium")
    plt.xlabel("$t$ (hours)")
    plt.ylabel("$N(t)$")
    plt.savefig("plot_transient.pdf")

def plot_vector_time(k,l,m,realizations,t_max):
    lb, average_T_vektor, t_list, average_n_list, iterations = simulate_N(k, l, m, realizations,t_max)
    x=(np.linspace(0, k,k+1))
    plt.plot(x, average_T_vektor /t_max)
    plt.title("Vector time - Verification of long term prob. ")
    plt.xlabel("$n$ jobs")
    plt.ylabel("$T_n / t_{max}$")
    plt.savefig("plot_vector_time.pdf")


