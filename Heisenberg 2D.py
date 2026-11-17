#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:22:13 2024

@author: josephglass
"""


import numpy as np
import matplotlib.pyplot as plt
#import numba
#from numba import njit
import random
import math
import statistics as stat

# Number of iterions to be used
N_iterations = 1000000

# Size of the lattice NxN
N = 10

# energy of system, beta * J
kT = 3
beta = 1/kT
J = 1
h = [0,0,1]


def createSpin():
    theta = random.uniform(0,2*math.pi)
    z = random.uniform(-1,1)
    x = math.sqrt(1-z**2)*math.cos(theta)
    y = math.sqrt(1-z**2)*math.sin(theta)
    coord = [x,y,z]
    return coord

'''
latticeUp = np.empty((N, N),dtype=object)
for k in range(N):
    for j in range(N):
        latticeUp[k][j] = tuple((0,0,1))
'''

latticeUp = np.zeros((N, N, 3))
for k in range(N):
    for j in range(N):
        latticeUp[k, j] = (0, 0, 1)
        
'''      
lattice75 = np.empty((N, N),dtype=object)
for k in range(N):
    for j in range(N):
        lattice75[k][j] = tuple((0,0,1))
prop = int(0.25*N*N)
for k in range(prop):
    check = list()
    same = True
    while same == True:
        x = random.randint(0,N-1)
        y = random.randint(0,N-1)
        if x not in check and y not in check:
            same = False
    lattice75[x][y] = createSpin()
'''  
  
lattice75 = np.zeros((N, N, 3))
for k in range(N):
    for j in range(N):
        lattice75[k, j] = (0, 0, 1)
prop = int(0.25*N*N)
for k in range(prop):
    check = list()
    same = True
    while same == True:
        x = random.randint(0,N-1)
        y = random.randint(0,N-1)
        if x not in check and y not in check:
            same = False
    lattice75[x][y] = createSpin()
    
    
'''
latticeR = np.empty((N,N),dtype=object)
for k in range(N):
    for j in range(N):
        latticeR[k][j] = createSpin()
'''       

latticeR = np.zeros((N, N, 3))
for k in range(N):
    for j in range(N):
        latticeR[k, j] = createSpin()
        

def get_energy(lattice):
    sumJ = 0
    termh = 0
    for y in range(N):
        for x in range(N):
            if y > 0: 
                sumJ += np.dot(lattice[x][y],lattice[x][y-1])
            if y < N-1:
                sumJ += np.dot(lattice[x][y],lattice[x][y+1])
            if x > 0:
                sumJ += np.dot(lattice[x][y],lattice[x-1][y])
            if x < N-1:
                sumJ += np.dot(lattice[x][y],lattice[x+1][y])
            termh += np.dot(h,lattice[x][y])
    energy = -J*sumJ - termh
    return energy
            
    

#@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nopython=True, nogil=True)
def metropolis(spin_arr, iterations, temp, energy):
    spin_arr2 = spin_arr.copy()
    
    net_spins = np.empty(iterations-1,dtype=object)
    net_spinsx = np.empty(iterations-1,dtype=object)
    net_spinsy = np.empty(iterations-1,dtype=object)
    net_spinsz = np.empty(iterations-1,dtype=object)
    
    net_energy = np.zeros(iterations-1)
    
    acceptance = list()
    
    for i in range(0,iterations-1):
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = spin_arr2[x][y] 
        spin_f = createSpin() 
        
        E_i = 0
        E_f = 0
        if x>0:
            E_i += -J*np.dot(spin_i,spin_arr2[x-1][y])
            E_f += -J*np.dot(spin_f,spin_arr2[x-1][y])
        if x<N-1:
            E_i += -J*np.dot(spin_i,spin_arr2[x+1][y])
            E_f += -J*np.dot(spin_f,spin_arr2[x+1][y])
        if y>0:
            E_i += -J*np.dot(spin_i,spin_arr2[x][y-1])
            E_f += -J*np.dot(spin_f,spin_arr2[x][y-1])
        if y<N-1:
            E_i += -J*np.dot(spin_i,spin_arr2[x][y+1])
            E_f += -J*np.dot(spin_f,spin_arr2[x][y+1])
            
        E_i += -np.dot(h,spin_i)
        E_f += -np.dot(h,spin_f)
            
        energy_Change = E_f-E_i

        if energy_Change<=0:
            spin_arr2[x][y]=spin_f
            energy += energy_Change
            acceptance.append(1)
              
        elif energy_Change>0 and random.random()<=math.exp(-energy_Change/temp):
            spin_arr2[x][y]=spin_f
            energy += energy_Change
            acceptance.append(1)
            
        else:
            acceptance.append(0)  
        
        '''              
        spinSum = [0,0,0]
        
        for k in range(N):
            for j in range(N):
                spinSum += np.add(spinSum,lattice2[k][j])
        
        spinSum = tuple(np.sum(lattice2))
        if t == 0:
            print(spinSum)
        '''      
        
        spinSum = np.sum(spin_arr2,axis=(0,1))
        
        net_spinsx[i] = spinSum[0]
        net_spinsy[i] = spinSum[1]
        net_spinsz[i] = spinSum[2]
        
        net_spins[i] = np.linalg.norm(spinSum)      
        net_energy[i] = energy
      
    #print(spinSum)
    #print(net_spinsx)    
    avgAcceptance = stat.mean(acceptance)
    print("Acceptance rate",avgAcceptance)
    
    return net_spins, net_energy, net_spinsx, net_spinsy, net_spinsz

spins,energies,spinsx,spinsy,spinsz = metropolis(latticeUp, N_iterations, kT, get_energy(latticeUp))

plt.figure(1)
plt.plot(spins/N**2)
plt.xlabel('Algorithm Time Steps')
plt.ylabel(r'Average Spin $\bar{M}$')
plt.grid()
plt.title('N=10x10, N_it=1,000,000, kT=0.1')
plt.show()

plt.figure(2)
plt.plot(energies)
plt.xlabel('Algorithm Time Steps')
plt.ylabel('Energy E')
plt.grid()
plt.title('N=10x10, N_it=1,000,000, kT=0.1')
plt.show()
#fig.tight_layout()
#fig.suptitle(r'Evolution of Average Spin and Energy for $\beta J=$0.7', y=1.07, size=18)



plt.figure(3)
plt.plot(spinsx/N**2,color = 'b',label='Mx')
plt.plot(spinsy/N**2,color = 'r',label='My')
plt.plot(spinsz/N**2,color = 'g',label='Mz')
plt.xlabel('Algorithm Time Steps')
plt.ylabel(r'Average Spin $\bar{M}$')
plt.title(r'Average Spin $\bar{M}$ of model')
plt.legend()
plt.grid()
plt.show()

''' 
def get_spin_energy(lattice, betas):
    ms = np.zeros(len(betas))
    E_means = np.zeros(len(betas))
    E_stds = np.zeros(len(betas))
    for i, beta in enumerate(betas):
        spins, energies = metropolis(lattice, N_iterations, beta, get_energy(lattice))
        ms[i] = spins[-100000:].mean()/N**2
        E_means[i] = energies[-100000:].mean()
        E_stds[i] = energies[-100000:].std()
    return ms, E_means, E_stds
   
betas = np.arange(0.1, 2, 0.1)

ms_Up, E_means_Up, E_stds_Up = get_spin_energy(latticeUp, betas)
#ms_R, E_means_R, E_stds_R = get_spin_energy(latticeR, betas)
ms_75, E_means_75, E_stds_75 = get_spin_energy(lattice75, betas)

plt.figure(figsize=(8,5))
plt.plot(betas, ms_Up, 'o--', label='Initial state of spins aligned upward')
#plt.plot(1/betas, ms_R, 'o--', label='Initial state of spins random')
plt.plot(betas, ms_75, 'o--', label='Initial state of spins ~75% algined up')
plt.xlabel('$kT$')
plt.ylabel(r'$\bar{M}$')
plt.legend(facecolor='white', framealpha=1)
plt.show()
'''
