"""Spread Option Pricing"""


import numpy as np
from random import gauss
from math import exp, sqrt, log
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import axes3d, Axes3D

def PriceByMC(t,T,S1_t,S2_t,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif):
    """ Calculation of the prace of a spread put/call european type option"""
    dt = T - t  # time to maturity
    drift1 = (r - q1 - 0.5*sigma1**2)*dt # drift term of asset 1
    vol1 = sigma1*sqrt(dt) # volatility term of asset 1
    drift2 = (r - q2 - 0.5*sigma2**2)*dt # drift term of asset 2
    vol2 = sigma2*sqrt(dt) # volatility term of asset 2    
    mean = 0.0
    std = 0.0
    for i in range(numOfSim):
        dB1=gauss(0.0,1.0) # BM 
        dB2=corr*dB1+sqrt(1-corr**2)*gauss(0.0,1.0) # another precess correlated to BM
        S1 = S1_t*exp(drift1 + vol1*dB1) # price of asset 1 corresponding to a particular path
        S2 = S2_t*exp(drift2 + vol2*dB2) # price of asset 2 corresponding to a particular path
        mean += max(optType*((S1-S2)*(1-modif)+abs(S1-S2)*modif-K),0) # recompute mean
        std += max(optType*((S1-S2)*(1-modif)+abs(S1-S2)*modif-K),0)*max(optType*((S1-S2)*(1-modif)+abs(S1-S2)*modif-K),0) # recompute variance
    mean *= (exp(-r*dt)/numOfSim) # mean price for numOfSim realizations
    std = sqrt((std/numOfSim - mean*mean)/numOfSim) # standard deviation for numOfSim realizations
    rslt = [mean,std]
    return rslt


"""Set the parameters"""

t=0 # time t
T=10 # maturity
S1_t=150 # price of asset 1 at time t
S2_t=100 # price of asset 2 at time t
K=50 # strike
r=0.05 # interest rate
q1=0.02 # dividend rate of asset 1 (=0 if assset does not yields any dividends)
q2=0.01 # dividend rate of asset 1 (=0 if assset does not yields any dividends)
sigma1=0.25 # volatility of the price of asset 1
sigma2=0.15 # volatility of the price of asset 2
corr=0.4 # correlation between the assets
optType=1 # type of the option: 1 if call, -1 if put
modif=0 # the model specification: 0 if standard spread option, 1 if modified

numOfSim=10000 # number of simulations
Price=PriceByMC(t,T,S1_t,S2_t,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)
print ("Price by MC simulations is  " + str(round(Price[0],2)))



""" Sensitivity analysis"""

"""Price sensitivity with respect to the asset prices - delta"""

n=10 # grid size 
P=np.empty([n, n])
S1 = np.linspace(0,S1_t*1.2,n) # grid 1
S2 = np.linspace(0,S2_t*1.2,n) # grid 2
for i in range(n):
    for j in range(n):
        P[i][j]=PriceByMC(t,T,S1[i],S2[j],K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]
y,x = np.meshgrid(S1,S2)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x, y, P, 100, cmap='viridis')
ax.set_xlabel('S_1')
ax.set_ylabel('S_2')
ax.set_zlabel('Price')
fig

dS1=S1_t*0.05 # price1 change
dS2=-S2_t*0.05 # price2 change
delta_1=(PriceByMC(t,T,S1_t+dS1,S2_t,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]-Price[0])/(dS1) # Δ(S1) - delta of asset 1
delta_2=(PriceByMC(t,T,S1_t,S2_t+dS2,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]-Price[0])/(dS2) # Δ(S2) - delta of asset 2
delta_spread=(PriceByMC(t,T,S1_t+dS1,S2_t+dS2,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]-Price[0])/(dS1-dS2) # Δ(S1-S2) - delta of spread

"""Further price sensitivity with respect to the asset prices - gamma"""

gamma_1=(PriceByMC(t,T,S1_t+dS1,S2_t,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]+PriceByMC(t,T,S1_t-dS1,S2_t,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]-2*Price[0])/(dS1**2) # Ɣ(S1) - gamma of asset 1
gamma_2=(PriceByMC(t,T,S1_t,S2_t+dS2,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]+PriceByMC(t,T,S1_t,S2_t-dS2,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]-2*Price[0])/(dS2**2) # Ɣ(S2) - gamma of asset 2
gamma_spread=(PriceByMC(t,T,S1_t+dS1,S2_t+dS2,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]+PriceByMC(t,T,S1_t-dS1,S2_t-dS2,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]-2*Price[0])/((dS1-dS2)**2) # Ɣ(S1-S2) - gamma of spread


"""Price sensitivity with respect to strike - kappa"""

m=100 # grid size 
kgrid = np.linspace(0,K*1.5,m) # grid
P_k=[PriceByMC(t,T,S1_t,S2_t,kgrid[i],r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0] for i in range(len(kgrid))]
plt.plot(kgrid.tolist(),P_k)
plt.xlabel('K')
plt.show()
dK=K*0.01 # strike change
kappa=(PriceByMC(t,T,S1_t,S2_t,K+dK,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]-Price[0])/(dK) # κ of the option


"""Price sensitivity with respect to time - theta"""

l=100 # grid size 
tgrid = np.linspace(0,T,l) # grid
P_t=[PriceByMC(tgrid[i],T,S1_t,S2_t,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0] for i in range(len(tgrid))]

plt.plot(tgrid.tolist(),P_t)
plt.xlabel('t')
plt.show()

d_t=1 # time change
theta=(PriceByMC(t+d_t,T,S1_t,S2_t,K,r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]-Price[0])/(d_t) # θ of the option


"""Price sensitivity with respect to interest rate - rho"""


p=100 # grid size
rgrid = np.linspace(0,r*3,p) # grid
P_r=[PriceByMC(t,T,S1_t,S2_t,K,rgrid[i],q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0] for i in range(len(rgrid))]

plt.plot(rgrid.tolist(),P_r)
plt.xlabel('r')
plt.show()

d_r=r*0.5 # interest rate change
rho=(PriceByMC(t,T,S1_t,S2_t,K,r+d_r,q1,q2,sigma1,sigma2,corr,numOfSim,optType,modif)[0]-Price[0])/(d_r) # ρ of the option

"""Price sensitivity with respect to volatility - vega"""

u=200 # grid size
sgrid = np.linspace(min(sigma1,sigma2)*0.5,max(sigma1,sigma2)*2,u) # grid
P_s1=[PriceByMC(t,T,S1_t,S2_t,K,r,q1,q2,sgrid[i],sigma2,corr,numOfSim,optType,modif)[0] for i in range(len(sgrid))]
P_s2=[PriceByMC(t,T,S1_t,S2_t,K,r,q1,q2,sigma1,sgrid[i],corr,numOfSim,optType,modif)[0] for i in range(len(sgrid))]

line1,=plt.plot(sgrid.tolist(),P_s1, label="asset 1")
line2,=plt.plot(sgrid.tolist(),P_s2, label="asset 2")
plt.xlabel('sigma')
first_legend = plt.legend(handles=[line1], loc=1)
ax = plt.gca().add_artist(first_legend)
plt.legend(handles=[line2], loc=5)
plt.show()

d_s=min(sigma1,sigma2)*0.1 # volatility change
vega1=(PriceByMC(t,T,S1_t,S2_t,K,r,q1,q2,sigma1+d_s,sigma2,corr,numOfSim,optType,modif)[0]-Price[0])/(d_s) # ν of asset 1
vega2=(PriceByMC(t,T,S1_t,S2_t,K,r,q1,q2,sigma1,sigma2+d_s,corr,numOfSim,optType,modif)[0]-Price[0])/(d_s) # ν of asset 2



"""Kirk's approach - for the case (modif=0) only!"""

sig=sqrt(sigma1**2+(sigma2*S2_t*exp(-q2*(T-t))/(S2_t*exp(-q2*(T-t))+K*exp(-r*(T-t))))**2-2*corr*sigma1*sigma2*(S2_t*exp(-q2*(T-t))/(S2_t*exp(-q2*(T-t))+K*exp(-r*(T-t))))) # volatility by Kirk
d1=(log(S1_t*exp(-q1*(T-t))/(S2_t*exp(-q2*(T-t))+K*exp(-r*(T-t))))+0.5*(sig**2)*(T-t))/(sig*sqrt(T-t))
d2=d1-sig*sqrt(T-t)
P_kirk=optType*(S1_t*exp(-q1*(T-t))*norm.cdf(optType*d1)-(S2_t*exp(-q2*(T-t))+K*exp(-r*(T-t)))*norm.cdf(optType*d2)) # price by Kirk's approach
print ("Price by Kirk's formula is  " + str(round(P_kirk,2)))








