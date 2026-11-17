#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:43:14 2024

@author: josephglass
"""

import numpy as np
import random
import math
import statistics as stat
import matplotlib.pyplot as plt

# Number of iterations to be used
N_iterations = 600000

# Size of the lattice N x N x N
N = 20

# Parameters
kT = 2
J = 1
h = [0,0,1]


# =============================================================================
# Define the function which uniformly chooses a random spin direction
# =============================================================================

def createSpin():
    theta = random.uniform(0,2*math.pi)
    phi = random.uniform(0,math.pi)
    z = math.cos(phi)
    x = math.sqrt(1-z**2)*math.cos(theta)
    y = math.sqrt(1-z**2)*math.sin(theta)
    coord = [x,y,z]
    return coord



# =============================================================================
# Create different inital lattices for study
# =============================================================================

latticeUp = np.zeros((N, N, N, 3))
for k in range(N):
    for j in range(N):
        for i in range(N):
            latticeUp[k,j,i] = (0, 0, 1)
        
lattice75 = np.zeros((N, N, N, 3))
for k in range(N):
    for j in range(N):
        for i in range(N):
            lattice75[k,j,i] = (0, 0, 1)
prop = int(0.25*(N**3))
for k in range(prop):
    checkx = list()
    checky = list()
    checkz = list()
    same = True
    while same == True:
        x = random.randint(0,N-1)
        y = random.randint(0,N-1)
        z = random.randint(0,N-1)
        if (x not in checkx)*(y not in checky)*(z not in checkz):
            same = False
    lattice75[x][y][z] = createSpin()
    checkx.append(x)
    checky.append(y)
    checkz.append(z)
       
lattice25 = np.zeros((N, N, N, 3))
for k in range(N):
    for j in range(N):
        for i in range(N):
            lattice25[k,j,i] = (0, 0, -1)
prop = int(0.25*(N**3))
for k in range(prop):
    checkx = list()
    checky = list()
    checkz = list()
    same = True
    while same == True:
        x = random.randint(0,N-1)
        y = random.randint(0,N-1)
        z = random.randint(0,N-1)
        if (x not in checkx)*(y not in checky)*(z not in checkz):
            same = False
    lattice25[x][y][z] = createSpin()
    checkx.append(x)
    checky.append(y)
    checkz.append(z)
         
latticeR = np.zeros((N, N, N, 3))
for k in range(N):
    for j in range(N):
        for i in range(N):
            latticeR[k,j,i] = createSpin()
        


# =============================================================================
# Function to find the energy of a given lattice
# =============================================================================

def E_lattice(lattice):
    sumJ = 0
    termh = 0
    for z in range(N):
        for y in range(N):
            for x in range(N):
                sumJ += np.dot(lattice[x,y,z],lattice[(x+1)%N,y,z])
                sumJ += np.dot(lattice[x,y,z],lattice[(x-1)%N,y,z])
                
                sumJ += np.dot(lattice[x,y,z],lattice[x,(y-1)%N,z])
                sumJ += np.dot(lattice[x,y,z],lattice[x,(y+1)%N,z])
                
                sumJ += np.dot(lattice[x,y,z],lattice[x,y,(z-1)%N])
                sumJ += np.dot(lattice[x,y,z],lattice[x,y,(z+1)%N])
            
                termh += np.dot(h,lattice[x][y][z])
    energy = -J*sumJ - termh
    return energy
        
    

# =============================================================================
# Metropolis algorithm    
# =============================================================================

def metropolis(lattice,iterations,temp,energy,field):
    lattice2 = lattice.copy()
    
    mag_Total = np.empty(iterations,dtype=object)
    mag_Totalx = np.empty(iterations,dtype=object)
    mag_Totaly = np.empty(iterations,dtype=object)
    mag_Totalz = np.empty(iterations,dtype=object)
    
    energy_Total = np.zeros(iterations)
    acceptance = list()
    
    for i in range(0,iterations):
        
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        z = np.random.randint(0,N)
        spin_Old = lattice2[x][y][z]
        spin_Trial = createSpin() 
        
        E_Old = 0
        E_Trial = 0
        
        E_Old += -J*np.dot(spin_Old,lattice2[(x-1)%N][y][z])
        E_Trial += -J*np.dot(spin_Trial,lattice2[(x-1)%N][y][z])
        
        E_Old += -J*np.dot(spin_Old,lattice2[(x+1)%N][y][z])
        E_Trial += -J*np.dot(spin_Trial,lattice2[(x+1)%N][y][z])
        
        E_Old += -J*np.dot(spin_Old,lattice2[x][(y-1)%N][z])
        E_Trial += -J*np.dot(spin_Trial,lattice2[x][(y-1)%N][z])
        
        E_Old += -J*np.dot(spin_Old,lattice2[x][(y+1)%N][z])
        E_Trial += -J*np.dot(spin_Trial,lattice2[x][(y+1)%N][z])
        
        E_Old += -J*np.dot(spin_Old,lattice2[x][y][(z-1)%N])
        E_Trial += -J*np.dot(spin_Trial,lattice2[x][y][(z-1)%N])
        
        E_Old += -J*np.dot(spin_Old,lattice2[x][y][(z+1)%N])
        E_Trial += -J*np.dot(spin_Trial,lattice2[x][y][(z+1)%N])
            
        E_Old += -np.dot(field,spin_Old)
        E_Trial += -np.dot(field,spin_Trial)
            
        energy_Change = E_Trial-E_Old

        if energy_Change<=0:
            lattice2[x][y][z]=spin_Trial
            energy += energy_Change
            acceptance.append(1)
            
        else:
            prob = math.exp(-energy_Change/temp)
            rnum = random.random()
            if rnum <= prob:
                lattice2[x][y][z]=spin_Trial
                energy += energy_Change
                acceptance.append(1)
            else:
                acceptance.append(0)
              
        spinSum = np.sum(lattice2,axis=(0,1,2))
        
        mag_Totalx[i] = spinSum[0]
        mag_Totaly[i] = spinSum[1]
        mag_Totalz[i] = spinSum[2]
        
        mag_Total[i] = np.linalg.norm(spinSum)      
        energy_Total[i] = energy
        '''
        if i == 20000:
            field = [0,0,0.75]
        if i == 40000:
            field = [0,0,0.50]
        if i == 600000:
            field = [0,0,0.25]
        if i == 80000:
            field = [0,0,0]
        '''
    #print(field)
    avgAcceptance = stat.mean(acceptance)
    print("Acceptance rate",avgAcceptance)
    
    return mag_Total, energy_Total, mag_Totalx, mag_Totaly, mag_Totalz,lattice2



# =============================================================================
# Looking at equilibration
# =============================================================================
'''
mag,energy,magx,magy,magz,outputLattice = metropolis(latticeR, N_iterations, kT, E_lattice(latticeR),h)

MC_step = [i/N**3 for i in range(N_iterations-1)]

plt.figure(1)
plt.plot(MC_step,mag/N**3)
plt.xlabel('Monte Carlo step')
plt.ylabel(r'Average magnetisation $\langle  |M| \rangle$')
plt.grid()
plt.show()

plt.figure(2)
plt.plot(MC_step,energy)
plt.xlabel('Monte Carlo step')
plt.ylabel('Energy E')
plt.grid()
plt.show()

plt.figure(3)
plt.plot(MC_step,magx/N**3,color = 'b',label=r'$\langle M_{x} \rangle$')
plt.plot(MC_step,magy/N**3,color = 'r',label=r'$\langle M_{y} \rangle$')
plt.plot(MC_step,magz/N**3,color = 'g',label=r'$\langle M_{z} \rangle$')
plt.xlabel('Monte Carlo step')
plt.ylabel(r'Average Magnetisation $\langle M_{i} \rangle$')
plt.legend()
plt.grid()
plt.show()
'''


# =============================================================================
# Temperature sweep of magnetisation
# =============================================================================
'''
def avg_Magnetisation(lattice, kT_range):
    mag_Means = np.zeros(len(kT_range))
    for i, temp in enumerate(kT_range):
        mag,energy,magx,magy,magz,outputLattice = metropolis(lattice, N_iterations, temp, E_lattice(lattice),h)
        mag_Means[i] = stat.mean(magz[-200000:])/N**3
    return mag_Means

kT_range = np.arange(0.1, 8.0, 0.1)

#mag_Means_Up = avg_Magnetisation(latticeUp, kT_range)
mag_Means_R = avg_Magnetisation(latticeR, kT_range)
mag_Means_75 = avg_Magnetisation(lattice75, kT_range)
#mag_Means_25 = avg_Magnetisation(lattice25, kT_range)

plt.figure(4)
plt.plot(kT_range,mag_Means_R,'o--',markersize=3,label='Initial state of spins random')
plt.plot(kT_range,mag_Means_75,'o--',markersize=3,label='Initial state of spins ~75% algined up')
plt.xlabel('$k_{B}T$',fontsize=12)
plt.ylabel(r'$\langle M_{z} \rangle$',fontsize=12)
plt.legend()
plt.grid()
plt.show()
'''


# =============================================================================
# Temperature sweep of heat capacity
# =============================================================================
'''
def energy_Variance(lattice, kT_range):
    energy_Vars = np.zeros(len(kT_range))
    
    for i, temp in enumerate(kT_range):
        mag,energy,magx,magy,magz,outputLattice = metropolis(lattice, N_iterations, temp, E_lattice(lattice),h)
        energy_Vars[i] = stat.variance(energy[-200000:])
        
    return energy_Vars

energy_Vars_R = energy_Variance(latticeR, kT_r-u;ange)
energy_Vars_75 = energy_Variance(lattice75, kT_range)

plt.figure(5)
plt.plot(kT_range,energy_Vars_75/((kT_range)*N**3), 'o--',markersize=3,label='Initial state of spins ~75% algined up')
plt.plot(kT_range,energy_Vars_R/((kT_range)*N**3), 'o--',markersize=3,label='Initial state of spins random')
plt.xlabel(r'$k_{B}T$',fontsize=12)
plt.ylabel(r'$C_{V}$',fontsize=12)
plt.legend()
plt.grid()
plt.show()
'''


# =============================================================================
# Getting the magnetic susceptibility
# =============================================================================

kT_range = np.arange(0.1, 8.0, 0.1)

def mag_Variance(lattice, kT_range):
    mag_Varsx = np.zeros(len(kT_range))
    mag_Varsy = np.zeros(len(kT_range))
    mag_Varsz = np.zeros(len(kT_range))
    
    for i, temp in enumerate(kT_range):
        mag,energy,magx,magy,magz,outputLattice = metropolis(lattice, N_iterations, temp, E_lattice(lattice),h)
        mag_Varsx[i] = stat.variance(magx[-200000:])/N**3
        mag_Varsy[i] = stat.variance(magy[-200000:])/N**3
        mag_Varsz[i] = stat.variance(magz[-200000:])/N**3
        
    return mag_Varsx,mag_Varsy,mag_Varsz

sus_x,sus_y,sus_z = mag_Variance(latticeR,kT_range)

plt.figure(7)
plt.plot(kT_range,sus_x/kT_range, 'o--',markersize=3,label='x-component')
plt.plot(kT_range,sus_y/kT_range, 'o--',markersize=3,label='y-component')
plt.plot(kT_range,sus_z/kT_range, 'o--',markersize=3,label='z-component')
plt.xlabel(r'$k_{B}T$',fontsize=12)
plt.ylabel(r'$\chi_{i}$',fontsize=12)
plt.grid()
plt.legend()
plt.show()



# =============================================================================
# Getting the fourth order cumulant
# =============================================================================
'''
### FOR N=20 ###

h=[0,0,0]

kT_range = np.arange(2.0, 3.0, 0.1)

def find_cumulant(lattice, kT_range):
    cumulant = np.zeros(len(kT_range))
    for i, temp in enumerate(kT_range):
        mag,energy,magx,magy,magz,outputLattice = metropolis(lattice, N_iterations, temp, E_lattice(lattice),h)
        mag_Means4 = stat.mean(magz[-200000:]**4)/N**3
        mag_Means2 = stat.mean(energy[-200000:]**2)/N**3
        cumulant[i] = 1 - ((1/3)*(mag_Means4/(mag_Means2**2)))
        
    return cumulant

cumulant20 = find_cumulant(latticeR, kT_range)

N = 10
N_iterations = 300000

latticeR = np.zeros((N, N, N, 3))
for k in range(N):
    for j in range(N):
        for i in range(N):
            latticeR[k,j,i] = createSpin()
       

cumulant10 = find_cumulant(latticeR, kT_range)

N = 30
N_iterations = 1500000

latticeR = np.zeros((N, N, N, 3))
for k in range(N):
    for j in range(N):
        for i in range(N):
            latticeR[k,j,i] = createSpin()
       

cumulant30 = find_cumulant(latticeR, kT_range)

plt.figure(6)
plt.plot(kT_range,cumulant10, 'o--',markersize=3,label='Lattice sidelength L=10')
plt.plot(kT_range,cumulant20, 'o--',markersize=3,label='Lattice sidelength L=20')
plt.plot(kT_range,cumulant30, 'o--',markersize=3,label='Lattice sidelength L=30')
plt.xlabel(r'$k_{B}T$',fontsize=12)
plt.ylabel(r'$U_{4}$',fontsize=12)
plt.legend()
plt.grid()
plt.show()
'''


# =============================================================================
# Hysteresis loop
# =============================================================================
'''
kT=0.1

N=10


latticeR = np.zeros((N, N, N, 3))
for k in range(N):
    for j in range(N):
        for i in range(N):
            latticeR[k,j,i] = createSpin()

def find_mag_at_h(lattice,field):
    h = field
    mag,energy,magx,magy,magz,outputLattice = metropolis(lattice, N_iterations, kT, E_lattice(lattice),h)
    mag_mean = stat.mean(magz[-300000:])/N**3
    return mag_mean,outputLattice


  

list_of_fields = [[0, 0, 0.0],[0, 0, 0.05],[0, 0, 0.1],[0, 0, 0.15],
 [0, 0, 0.2],[0, 0, 0.25],[0, 0, 0.3],[0, 0, 0.35],[0, 0, 0.4],[0, 0, 0.45],
 [0, 0, 0.5],[0, 0, 0.55],[0, 0, 0.6],[0, 0, 0.65],[0, 0, 0.7],[0,0,0.75],
 [0, 0, 0.8],[0, 0, 0.85],[0, 0, 0.9],[0, 0, 0.95],[0, 0, 1.0],[0, 0, 0.95],
 [0, 0, 0.9],[0, 0, 0.85],[0, 0, 0.8],[0, 0, 0.75],[0, 0, 0.7],[0, 0, 0.65],
 [0, 0, 0.6],[0, 0, 0.55],[0, 0, 0.5],[0, 0, 0.45],[0, 0, 0.40],[0, 0, 0.35],
 [0, 0, 0.30],[0, 0, 0.25],[0, 0, 0.20],[0, 0, 0.15],[0, 0, 0.1],[0, 0, 0.05],
 [0, 0, 0.0],[0, 0, -0.05],[0, 0, -0.1],[0, 0, -0.15],[0, 0, -0.2],
 [0, 0, -0.25],[0, 0, -0.3],[0, 0, -0.35],[0, 0, -0.4],[0, 0, -0.45],
 [0, 0, -0.5],[0, 0, -0.55],[0, 0, -0.6],[0, 0, -0.65],[0,0,-0.7],
 [0, 0, -0.75],
 [0, 0, -0.8],[0, 0, -0.85],[0, 0, -0.9],[0, 0, -0.95],[0, 0, -1.0],
 [0, 0, -0.95],[0, 0, -0.9],[0, 0, -0.85],[0, 0, -0.8],[0, 0, -0.75],
 [0, 0, -0.7],[0, 0, -0.65],[0, 0, -0.6],[0, 0, -0.55],[0, 0, -0.5],
 [0, 0, -0.45],[0, 0, -0.4],[0, 0, -0.35],
 [0, 0, -0.3],[0, 0, -0.25],[0, 0, -0.2],[0, 0, -0.15],[0, 0, -0.1],
 [0, 0, -0.05],[0, 0, 0.0],[0, 0, 0.05],[0, 0, 0.1],[0, 0, 0.15],[0, 0, 0.2],
 [0, 0, 0.25],[0, 0, 0.3],[0, 0, 0.35],[0, 0, 0.4],
 [0, 0, 0.45],[0, 0, 0.5],[0, 0, 0.55],[0, 0, 0.6],[0, 0, 0.65],[0, 0, 0.7],
 [0, 0, 0.75],[0, 0, 0.8],[0, 0, 0.85],[0, 0, 0.9],[0, 0, 0.95],[0, 0, 1.0]]        
       
hystersis_mags = list()
newLattice = latticeR

for k in range(len(list_of_fields)):
    mag_at_h,newLattice=find_mag_at_h(newLattice,list_of_fields[k])
    hystersis_mags.append(mag_at_h)
   
   
plot_field_list = list()
for k in range(len(list_of_fields)):
    value = sum(list_of_fields[k])
    plot_field_list.append(value)

   
plt.figure(8)
plt.plot(plot_field_list,hystersis_mags)
plt.xlabel(r'$\mathbf{h}$',fontsize=12)
plt.ylabel(r'$\langle M_{z} \rangle$',fontsize=12)
plt.grid()
plt.show()
'''


# =============================================================================
# abs mag plot against h
# =============================================================================
'''
kT = 1

magVals = list()

list_of_fields = [[0,0,0+i/10] for i in range(11)]

for k in range(len(list_of_fields)):
    mag,energy,magx,magy,magz,outputLattice=metropolis(latticeR,
            N_iterations,kT,E_lattice(latticeR),list_of_fields[k])
    mag = stat.mean(mag[-300000:])/N**3
    magVals.append(mag)

plot_field_list = list()
for k in range(len(list_of_fields)):
    value = sum(list_of_fields[k])
    plot_field_list.append(value)

plt.figure(9)
plt.plot(plot_field_list,magVals)
plt.xlabel(r'$h_{z}$')
plt.ylabel(r'$\langle | \mathbf{M} | \rangle$')
plt.grid()
plt.show()
'''

