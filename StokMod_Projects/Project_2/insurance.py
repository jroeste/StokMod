import numpy as np

n=100000 #Number of years
lam=3;t=59;mu=-2;sigma=1; 
C=[0]*n


for i in range(0,n):
    N=np.random.poisson(lam=lam*t, size=1) #Sample the number of insurance claims from the poisson distriution.
    C[i]=sum(np.random.lognormal(mean=mu, sigma=sigma, size=N)) #Sample the insurance amount claims from the lognormal distribution.

print("Expected value: ", np.mean(C))
print("Variance: " ,np.var(C))
