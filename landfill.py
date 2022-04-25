#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:25:32 2022

@author: wenyu
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import MyTicToc as mt
import pandas as pd


    
#Scl-storage in cover layer; Swb-storage in the waste layer; Sdr-drainage lyer
# R is the rainfall rate, E is the evaporation rate 
data= pd.read_csv('WieringermeerData_Meteo.csv', header = 0, index_col = 0, parse_dates=True)
data.interpolate(method = 'linear', inplace=True)
R = data.iloc[:,0].values
pEV = data.iloc[:,1].values
temp = data.iloc[:,2].values

    
a = 0.1
p = 0.4 #porosity
bcl = 8
bwb = 20
beta0 = 0.2
Cf = 1
Sclmin = 0
Sclmax = 1.5*p
Swbmin = 0
Swbmax = 12*p
Sevmin = 0
Sevmax = 0.006
dt = 0.1
    

## landfill model

def dsdt(t, Y):
    Scl = Y[0]
    Swb = Y[1]

    Lcl = a * ((Scl-Sclmin)/(Sclmax-Sclmin))**bcl
    Lwb = a * ((Swb-Swbmin)/(Swbmax-Swbmin))**bwb

    if Scl < Sevmin:
        fred = 0
    elif (Scl>=Sevmin) & (Scl<=Sevmax):
        fred = (Scl-Sevmin)/(Sevmax-Sevmin)
    else:
        fred = 1
 
    E = pEV[int(t)] * Cf * fred
    beta = beta0 * ((Scl-Sclmin)/(Sclmax-Sclmin))
    #Qdr = beta * Lcl + Lwb
    dScldt = R[int(t)] - Lcl - E
    dSwbdt = (1-beta)*Lcl - Lwb
    #dSdrdt = beta * Lcl + Lwb - Qdr
   
    return np.array([dScldt, dSwbdt])


def main():
        # Definition of output times
    tOut = np.arange(0, len(data.iloc[:,0]), 1)             # time
    nOut = np.shape(tOut)[0]
    Y0 = np.array([0.06, 0.5]) #initial Scl,Swb,Sdr
    #mt.tic()
    t_span = [tOut[0], tOut[-1]]
    YODE = scipy.integrate.solve_ivp(dsdt, t_span, Y0, t_eval=tOut, method='RK45', vectorized=True, rtol=1e-5 )
    # infodict['message']                     # >>> 'Integration successful.'
    SCL = YODE.y[0,:]
    SWB = YODE.y[1,:]
    #Qdr = YODE.y[2,:]
    Scl = np.zeros(len(SCL))
    Swb = np.zeros(len(SCL))
    #Sdr = np.zeros(len(SCL))
    
    
    
    for i in range(0, len(SCL)-1):
        Scl[i] = SCL[i+1]-SCL[i]
        Swb[i] = SWB[i+1]-SWB[i]
        #Sdr[i] = SDR[i+1]-SDR[i]
 
        
 
        
        '''Eular'''     
    
    YEuler = np.zeros([nOut, 2], dtype=float)
    SclEuler = np.zeros(nOut)
    SwbEuler = np.zeros(nOut)
    #QdrEuler = np.zeros(nOut)

    dtMax = 0.1
    # dtMin = 1e-11
    t = tOut[0]
    iiOut = 0

    # Initialize problem
    mt.tic()
    Y = Y0
    # Write initial values to output vector
    YEuler[iiOut, [0, 1]] = Y
    
    while (t < tOut[nOut-1]):
        # check time steps
        Rates = dsdt(t, Y)
        dtRate = -0.7 * Y/(Rates + 1e-18) #maximum 70% decrease
        dtOut = tOut[iiOut+1]-t
        dtRate = (dtRate <= 0)*dtMax + (dtRate > 0)*dtRate
        dt = min(min(dtRate), dtOut, dtMax)

        Y = Y + Rates * dt
        t = t + dt

        # print ("Time to Output is " + str(np.abs(tOut[iiOut+1]-t)) +
        # " days.")

        if (np.abs(tOut[iiOut+1]-t) < 1e-5):
            YEuler[iiOut+1, [0, 1]] = Y
            iiOut += 1

    SCLEuler, SWBEuler = YEuler.T
    mt.toc()
    
    for i in range(0, len(SwbEuler)-1):
        SclEuler[i] = SCLEuler[i+1]-SCLEuler[i]
        SwbEuler[i] = SWBEuler[i+1]-SWBEuler[i]
        
    

    '''Plot'''
    plt.figure()
    plt.plot(tOut, Scl, 'r-', label='Scl')
    plt.plot(tOut, Swb, 'b-', label='Swb')
   # plt.plot(tOut, Sdr, 'g-', label='Sdr')
    plt.legend()
    
    
    
    plt.figure()
    plt.plot(tOut, SclEuler, 'r-', label='SclEuler')
    plt.plot(tOut, SwbEuler, 'b-', label='SwbEuler')
   
    plt.legend()
   
    
    

    
    

if __name__ == "__main__":
    main()
   
    
    
    
    
    
    
    
    
    
    
    




