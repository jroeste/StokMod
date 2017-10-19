import numpy as np
import numpy

t=59; #Number of days
lamda=3;
mu=-2;
sigma=1; 
C = [0]*t;


for i in range(0,t):
    N = numpy.random.poisson(lam=lamda, size=1); #Sample the number of insurance claims from the poisson distriution.
    Z[i] = sum(numpy.random.lognormal(mean=mu, sigma=sigma, size=N)); #Sample the insurance amount claims from the lognormal distribution.

print("Expected value: ", numpy.mean(C));
print("Variance: " ,numpy.var(C));