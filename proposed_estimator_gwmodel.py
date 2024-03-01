# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:38:47 2020

@author: Jing

"""
#%% Import modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from SALib.sample import saltelli

#%% Global model settings
    
seed = 1024                  # Random seed
np.random.seed(seed)
plt.style.use('default')

# Model info
Ma, Mb, Mc = 2, 2, 2         # Number of process models 
PMA = np.array([0.5, 0.5])   # Process model weights
PMB = np.array([0.5, 0.5])
PMC = np.array([0.5, 0.5])

# Parameters for snow melt process
P = 60                       # Precipation in inch/yr
Ta = 7                       # Average temperature for a given day in degree 
Tm = 0                       # Critical snow melt point in degree
Csn = 0.8                    # Runoff confficient
SVC = 0.7                    # Snow cover fraction 
A = 2000 * 1e6               # Upper catchment area in  km2
Rn = 80                      # Surface radiation in w/m2

# Boundary condition
h1 = 300                     # Head in the left 

# Domain information
z0 = 289                     # Elevation of river bed in meters    
L = 10000                    # Length of modeling domain
x0 = 7000                    
nx = 21
xs = np.linspace(0, L, nx, endpoint=True)

# Parameter bounds and distributions
bounds = {'a' : [2.0, 0.4],
          'b' : [0.2, 0.5],
          'hk': [2.9, 0.5],
          'k1': [2.6, 0.3],
          'k2': [3.2, 0.3],
          'f1': [3.5, 0.75],
          'f2': [2.5, 0.3],
          'r' : [0.3, 0.05]}

dists = {'a' : 'norm',
         'b' : 'unif',
         'hk': 'lognorm',
         'k1': 'lognorm',
         'k2': 'lognorm',
         'f1': 'norm',
         'f2': 'norm',
         'r' : 'norm'}

# Problem defination
problem = {'num_vars': 8,
           'names': ['a', 'b', 'hk', 'k1', 'k2', 'f1', 'f2', 'r'],
           'bounds': [bounds['a'], bounds['b'], bounds['hk'], bounds['k1'], 
                      bounds['k2'], bounds['f1'], bounds['f2'], bounds['r']],
           'dists': [dists['a'], dists['b'], dists['hk'], dists['k1'], 
                     dists['k2'], dists['f1'], dists['f2'], dists['r']]
           }

#%% Process model functions
def model_R1(a):
    """
    Compute recharge[m/d] using recharge model R1 by Chaturvedi(1936)
    
    """
    return a * (P - 14)**0.5 * 25.4 * 0.001 / 365

def model_R2(b):
    """
    Compute recharge[m/d] using recharge model R2 by Krishna Rao (1970)
    
    """
    return b * (P - 15.7) * 25.4 * 0.001 / 365

def model_G1(hk):
    """
    Homogenous condition 
    
    """
    return hk, hk

def model_G2(k1, k2):
    """
    heterogenous condition 
    
    """
    return k1, k2

def model_M1(f1):
    """
    Compute river stage h2 [m] using degree-day method
 
    """
    M = f1 * (Ta - Tm)
    Q = Csn * M * SVC * A * 0.001 / 86400
    h2 = 0.3 * Q**0.6 + z0
    
    return h2

def model_M2(f2, r):
    """
    Compute river stage h2 [m] using restricted degree-day radiation balance approach

    """
    M = f2 * (Ta - Tm) + r * Rn
    Q = Csn * M * SVC * A * 0.001 / 86400
    h2 = 0.3 * Q**0.6 + z0
    
    return h2

#%% System model function
def cmpt_Y(x, w, k1, k2, h2, ):
    """
    Compute discharge per unit [m2/d] at a given location x using anaytical solution
    
    """
    C1 = (h1**2 - h2**2 - w / k1 * x0**2 + w / k2 * x0**2 - w / k2 * L**2) / (k1 / k2 * x0 - k1 / k2 * L - x0)

    return w * x - k1 * C1 / 2

#%% Parameter generation
def generate_param_values(N):
    """
    Generate parameter matrices A and B using SALib.saltelli.sample
    Details for this fuction can be found online.

    """
    A = saltelli.sample(problem, N, 
                        calc_second_order=False)[::problem['num_vars'] + 2, :]
    B = saltelli.sample(problem, N, 
                        calc_second_order=False)[problem['num_vars'] + 1::problem['num_vars'] + 2, :]
    
    return A, B
    
#%% Estimator function
def fast_MC(x, pvalues_A, pvalues_B, print_to_console=False):
    """
    Parameters
    ----------
    x         : the x-coordinate of the intrested location
        Used to compute the model output.
    pvalues_A : Two dimensional sample matrix.
        Used to generate matrix A.
    pvalues_B : Two dimensional sample matrix.
        Used to generate matrix B.

    Returns
    -------
    ST_A : float
        Total-process sensitivity index of rechagre process.
    ST_B : float
        Total-process sensitivity index of geology process.
    ST_C : float
        Total-process sensitivity index of snowmelt process.

    """  
    # ================================ R1G1M1 =================================
    # Matrix A
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M1(pvalues_A[:, 5])
    Y_A_R1G1M1  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix B
    w           = model_R1(pvalues_B[:, 0])
    k1, k2      = model_G1(pvalues_B[:, 2])
    h2          = model_M1(pvalues_B[:, 5])
    Y_B_R1G1M1  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C1
    w           = model_R1(pvalues_B[:, 0])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M1(pvalues_A[:, 5])
    Y_C1_R1G1M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C2
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G1(pvalues_B[:, 2])
    h2          = model_M1(pvalues_A[:, 5])
    Y_C2_R1G1M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C3
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M1(pvalues_B[:, 5])
    Y_C3_R1G1M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Parameter sensitivity indices; Saltelli (2010). COMPUT PHYS COMMUN.
    Si_A = np.mean(Y_B_R1G1M1 * (Y_C1_R1G1M1 - Y_A_R1G1M1)) / np.var(Y_A_R1G1M1)
    Si_B = np.mean(Y_B_R1G1M1 * (Y_C2_R1G1M1 - Y_A_R1G1M1)) / np.var(Y_A_R1G1M1)
    Si_C = np.mean(Y_B_R1G1M1 * (Y_C3_R1G1M1 - Y_A_R1G1M1)) / np.var(Y_A_R1G1M1)

    ST_A = np.mean(Y_A_R1G1M1**2 - Y_A_R1G1M1 * Y_C1_R1G1M1) / np.var(Y_A_R1G1M1)
    ST_B = np.mean(Y_A_R1G1M1**2 - Y_A_R1G1M1 * Y_C2_R1G1M1) / np.var(Y_A_R1G1M1)
    ST_C = np.mean(Y_A_R1G1M1**2 - Y_A_R1G1M1 * Y_C3_R1G1M1) / np.var(Y_A_R1G1M1)
    
    if print_to_console:
        print('\nParameter sensitivity of system model R1G1M1')
        print('Si_A = %.4f Si_B = %.4f Si_C = %.4f' %(Si_A, Si_B, Si_C))
        print('ST_A = %.4f ST_B = %.4f ST_C = %.4f\n' %(ST_A, ST_B, ST_C))
    
    # ==============================R1G1M2 ==============================
    # Matrix A
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_A_R1G1M2  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix B
    w           = model_R1(pvalues_B[:, 0])
    k1, k2      = model_G1(pvalues_B[:, 2])
    h2          = model_M2(pvalues_B[:, 6], pvalues_B[:, 7])
    Y_B_R1G1M2  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C1
    w           = model_R1(pvalues_B[:, 0])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_C1_R1G1M2 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C2
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G1(pvalues_B[:, 2])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_C2_R1G1M2 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C3
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M2(pvalues_B[:, 6], pvalues_B[:, 7])
    Y_C3_R1G1M2 = cmpt_Y(x, w, k1, k2, h2)
    
    # Parameter sensitivity indices; Saltelli (2010). COMPUT PHYS COMMUN.
    Si_A = np.mean(Y_B_R1G1M2 * (Y_C1_R1G1M2 - Y_A_R1G1M2)) / np.var(Y_A_R1G1M2)
    Si_B = np.mean(Y_B_R1G1M2 * (Y_C2_R1G1M2 - Y_A_R1G1M2)) / np.var(Y_A_R1G1M2)
    Si_C = np.mean(Y_B_R1G1M2 * (Y_C3_R1G1M2 - Y_A_R1G1M2)) / np.var(Y_A_R1G1M2)

    ST_A = np.mean(Y_A_R1G1M2**2 - Y_A_R1G1M2 * Y_C1_R1G1M2) / np.var(Y_A_R1G1M2)
    ST_B = np.mean(Y_A_R1G1M2**2 - Y_A_R1G1M2 * Y_C2_R1G1M2) / np.var(Y_A_R1G1M2)
    ST_C = np.mean(Y_A_R1G1M2**2 - Y_A_R1G1M2 * Y_C3_R1G1M2) / np.var(Y_A_R1G1M2)
    
    if print_to_console:
        print('\nParameter sensitivity of system model R1G1M2')
        print('Si_A = %.4f Si_B = %.4f Si_C = %.4f' %(Si_A, Si_B, Si_C))
        print('ST_A = %.4f ST_B = %.4f ST_C = %.4f\n' %(ST_A, ST_B, ST_C))
    
    # ==============================R1G2M1 ==============================
    # Matrix A
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M1(pvalues_A[:, 5])
    Y_A_R1G2M1  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix B
    w           = model_R1(pvalues_B[:, 0])
    k1, k2      = model_G2(pvalues_B[:, 3], pvalues_B[:, 4])
    h2          = model_M1(pvalues_B[:, 5])
    Y_B_R1G2M1  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C1
    w           = model_R1(pvalues_B[:, 0])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M1(pvalues_A[:, 5])
    Y_C1_R1G2M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C2
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G2(pvalues_B[:, 3], pvalues_B[:, 4])
    h2          = model_M1(pvalues_A[:, 5])
    Y_C2_R1G2M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C3
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M1(pvalues_B[:, 5])
    Y_C3_R1G2M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Parameter sensitivity indices; Saltelli (2010). COMPUT PHYS COMMUN.
    Si_A = np.mean(Y_B_R1G2M1 * (Y_C1_R1G2M1 - Y_A_R1G2M1)) / np.var(Y_A_R1G2M1)
    Si_B = np.mean(Y_B_R1G2M1 * (Y_C2_R1G2M1 - Y_A_R1G2M1)) / np.var(Y_A_R1G2M1)
    Si_C = np.mean(Y_B_R1G2M1 * (Y_C3_R1G2M1 - Y_A_R1G2M1)) / np.var(Y_A_R1G2M1)

    ST_A = np.mean(Y_A_R1G2M1**2 - Y_A_R1G2M1 * Y_C1_R1G2M1) / np.var(Y_A_R1G2M1)
    ST_B = np.mean(Y_A_R1G2M1**2 - Y_A_R1G2M1 * Y_C2_R1G2M1) / np.var(Y_A_R1G2M1)
    ST_C = np.mean(Y_A_R1G2M1**2 - Y_A_R1G2M1 * Y_C3_R1G2M1) / np.var(Y_A_R1G2M1)
    
    if print_to_console:
        print('\nParameter sensitivity of system model R1G2M1')
        print('Si_A = %.4f Si_B = %.4f Si_C = %.4f' %(Si_A, Si_B, Si_C))
        print('ST_A = %.4f ST_B = %.4f ST_C = %.4f\n' %(ST_A, ST_B, ST_C))
    
    # ==============================R1G2M2 ==============================
    # Matrix A
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_A_R1G2M2  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix B
    w           = model_R1(pvalues_B[:, 0])
    k1, k2      = model_G2(pvalues_B[:, 3], pvalues_B[:, 4])
    h2          = model_M2(pvalues_B[:, 6], pvalues_B[:, 7])
    Y_B_R1G2M2  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C1
    w           = model_R1(pvalues_B[:, 0])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_C1_R1G2M2 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C2
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G2(pvalues_B[:, 3], pvalues_B[:, 4])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_C2_R1G2M2 = cmpt_Y(x, w, k1, k2, h2)

    
    # Matrix C3
    w           = model_R1(pvalues_A[:, 0])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M2(pvalues_B[:, 6], pvalues_B[:, 7])
    Y_C3_R1G2M2 = cmpt_Y(x, w, k1, k2, h2)
    
    # Parameter sensitivity indices; Saltelli (2010). COMPUT PHYS COMMUN.
    Si_A = np.mean(Y_B_R1G2M2 * (Y_C1_R1G2M2 - Y_A_R1G2M2)) / np.var(Y_A_R1G2M2)
    Si_B = np.mean(Y_B_R1G2M2 * (Y_C2_R1G2M2 - Y_A_R1G2M2)) / np.var(Y_A_R1G2M2)
    Si_C = np.mean(Y_B_R1G2M2 * (Y_C3_R1G2M2 - Y_A_R1G2M2)) / np.var(Y_A_R1G2M2)

    ST_A = np.mean(Y_A_R1G2M2**2 - Y_A_R1G2M2 * Y_C1_R1G2M2) / np.var(Y_A_R1G2M2)
    ST_B = np.mean(Y_A_R1G2M2**2 - Y_A_R1G2M2 * Y_C2_R1G2M2) / np.var(Y_A_R1G2M2)
    ST_C = np.mean(Y_A_R1G2M2**2 - Y_A_R1G2M2 * Y_C3_R1G2M2) / np.var(Y_A_R1G2M2)
    
    if print_to_console:
        print('\nParameter sensitivity of system model R1G2M2')
        print('Si_A = %.4f Si_B = %.4f Si_C = %.4f' %(Si_A, Si_B, Si_C))
        print('ST_A = %.4f ST_B = %.4f ST_C = %.4f\n' %(ST_A, ST_B, ST_C))
    
    # ================================ R2G1M1 =================================
    # Matrix A
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M1(pvalues_A[:, 5])
    Y_A_R2G1M1  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix B
    w           = model_R2(pvalues_B[:, 1])
    k1, k2      = model_G1(pvalues_B[:, 2])
    h2          = model_M1(pvalues_B[:, 5])
    Y_B_R2G1M1  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C1
    w           = model_R2(pvalues_B[:, 1])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M1(pvalues_A[:, 5])
    Y_C1_R2G1M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C2
    w           = model_R2( pvalues_A[:, 1])
    k1, k2      = model_G1(  pvalues_B[:, 2])
    h2          = model_M1( pvalues_A[:, 5])
    Y_C2_R2G1M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C3
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M1(pvalues_B[:, 5])
    Y_C3_R2G1M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Parameter sensitivity indices; Saltelli (2010). COMPUT PHYS COMMUN.
    Si_A = np.mean(Y_B_R2G1M1 * (Y_C1_R2G1M1 - Y_A_R2G1M1)) / np.var(Y_A_R2G1M1)
    Si_B = np.mean(Y_B_R2G1M1 * (Y_C2_R2G1M1 - Y_A_R2G1M1)) / np.var(Y_A_R2G1M1)
    Si_C = np.mean(Y_B_R2G1M1 * (Y_C3_R2G1M1 - Y_A_R2G1M1)) / np.var(Y_A_R2G1M1)

    ST_A = np.mean(Y_A_R2G1M1**2 - Y_A_R2G1M1 * Y_C1_R2G1M1) / np.var(Y_A_R2G1M1)
    ST_B = np.mean(Y_A_R2G1M1**2 - Y_A_R2G1M1 * Y_C2_R2G1M1) / np.var(Y_A_R2G1M1)
    ST_C = np.mean(Y_A_R2G1M1**2 - Y_A_R2G1M1 * Y_C3_R2G1M1) / np.var(Y_A_R2G1M1)
    
    if print_to_console:
        print('\nParameter sensitivity of system model R2G1M1')
        print('Si_A = %.4f Si_B = %.4f Si_C = %.4f' %(Si_A, Si_B, Si_C))
        print('ST_A = %.4f ST_B = %.4f ST_C = %.4f\n' %(ST_A, ST_B, ST_C))
    
    # =============================== R2G1M2 ==================================
    # Matrix A
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_A_R2G1M2  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix B
    w           = model_R2(pvalues_B[:, 1])
    k1, k2      = model_G1(pvalues_B[:, 2])
    h2          = model_M2(pvalues_B[:, 6], pvalues_B[:, 7])
    Y_B_R2G1M2  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C1
    w           = model_R2(pvalues_B[:, 1])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_C1_R2G1M2 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C2
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G1(pvalues_B[:, 2])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_C2_R2G1M2 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C3
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G1(pvalues_A[:, 2])
    h2          = model_M2(pvalues_B[:, 6], pvalues_B[:, 7])
    Y_C3_R2G1M2 = cmpt_Y(x, w, k1, k2, h2)
    
    # Parameter sensitivity indices; Saltelli (2010). COMPUT PHYS COMMUN.
    Si_A = np.mean(Y_B_R2G1M2 * (Y_C1_R2G1M2 - Y_A_R2G1M2)) / np.var(Y_A_R2G1M2)
    Si_B = np.mean(Y_B_R2G1M2 * (Y_C2_R2G1M2 - Y_A_R2G1M2)) / np.var(Y_A_R2G1M2)
    Si_C = np.mean(Y_B_R2G1M2 * (Y_C3_R2G1M2 - Y_A_R2G1M2)) / np.var(Y_A_R2G1M2)

    ST_A = np.mean(Y_A_R2G1M2**2 - Y_A_R2G1M2 * Y_C1_R2G1M2) / np.var(Y_A_R2G1M2)
    ST_B = np.mean(Y_A_R2G1M2**2 - Y_A_R2G1M2 * Y_C2_R2G1M2) / np.var(Y_A_R2G1M2)
    ST_C = np.mean(Y_A_R2G1M2**2 - Y_A_R2G1M2 * Y_C3_R2G1M2) / np.var(Y_A_R2G1M2)
    
    if print_to_console:
        print('\nParameter sensitivity of system model R2G1M2')
        print('Si_A = %.4f Si_B = %.4f Si_C = %.4f' %(Si_A, Si_B, Si_C))
        print('ST_A = %.4f ST_B = %.4f ST_C = %.4f\n' %(ST_A, ST_B, ST_C))
    
    # ================================ R2G2M1 =================================
    # Matrix A
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M1(pvalues_A[:, 5])
    Y_A_R2G2M1  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix B
    w           = model_R2(pvalues_B[:, 1])
    k1, k2      = model_G2(pvalues_B[:, 3], pvalues_B[:, 4])
    h2          = model_M1(pvalues_B[:, 5])
    Y_B_R2G2M1  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C1
    w           = model_R2(pvalues_B[:, 1])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M1(pvalues_A[:, 5])
    Y_C1_R2G2M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C2
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G2(pvalues_B[:, 3], pvalues_B[:, 4])
    h2          = model_M1(pvalues_A[:, 5])
    Y_C2_R2G2M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C3
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M1(pvalues_B[:, 5])
    Y_C3_R2G2M1 = cmpt_Y(x, w, k1, k2, h2)
    
    # Parameter sensitivity indices; Saltelli (2010). COMPUT PHYS COMMUN.
    Si_A = np.mean(Y_B_R2G2M1 * (Y_C1_R2G2M1 - Y_A_R2G2M1)) / np.var(Y_A_R2G2M1)
    Si_B = np.mean(Y_B_R2G2M1 * (Y_C2_R2G2M1 - Y_A_R2G2M1)) / np.var(Y_A_R2G2M1)
    Si_C = np.mean(Y_B_R2G2M1 * (Y_C3_R2G2M1 - Y_A_R2G2M1)) / np.var(Y_A_R2G2M1)

    ST_A = np.mean(Y_A_R2G2M1**2 - Y_A_R2G2M1 * Y_C1_R2G2M1) / np.var(Y_A_R2G2M1)
    ST_B = np.mean(Y_A_R2G2M1**2 - Y_A_R2G2M1 * Y_C2_R2G2M1) / np.var(Y_A_R2G2M1)
    ST_C = np.mean(Y_A_R2G2M1**2 - Y_A_R2G2M1 * Y_C3_R2G2M1) / np.var(Y_A_R2G2M1)
    
    if print_to_console:
        print('\nParameter sensitivity of system model R2G2M1')
        print('Si_A = %.4f Si_B = %.4f Si_C = %.4f' %(Si_A, Si_B, Si_C))
        print('ST_A = %.4f ST_B = %.4f ST_C = %.4f\n' %(ST_A, ST_B, ST_C))
    
    # ================================ R2G2M2 =================================
    # Matrix A
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_A_R2G2M2  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix B
    w           = model_R2(pvalues_B[:, 1])
    k1, k2      = model_G2(pvalues_B[:, 3], pvalues_B[:, 4])
    h2          = model_M2(pvalues_B[:, 6], pvalues_B[:, 7])
    Y_B_R2G2M2  = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C1
    w           = model_R2(pvalues_B[:, 1])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_C1_R2G2M2 = cmpt_Y(x, w, k1, k2, h2)
    
    # Matrix C2
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G2(pvalues_B[:, 3], pvalues_B[:, 4])
    h2          = model_M2(pvalues_A[:, 6], pvalues_A[:, 7])
    Y_C2_R2G2M2 = cmpt_Y(x, w, k1, k2, h2)

    
    # Matrix C3
    w           = model_R2(pvalues_A[:, 1])
    k1, k2      = model_G2(pvalues_A[:, 3], pvalues_A[:, 4])
    h2          = model_M2(pvalues_B[:, 6], pvalues_B[:, 7])
    Y_C3_R2G2M2 = cmpt_Y(x, w, k1, k2, h2)
    
    # Parameter sensitivity indices; Saltelli (2010). COMPUT PHYS COMMUN.
    Si_A = np.mean(Y_B_R2G2M2 * (Y_C1_R2G2M2 - Y_A_R2G2M2)) / np.var(Y_A_R2G2M2)
    Si_B = np.mean(Y_B_R2G2M2 * (Y_C2_R2G2M2 - Y_A_R2G2M2)) / np.var(Y_A_R2G2M2)
    Si_C = np.mean(Y_B_R2G2M2 * (Y_C3_R2G2M2 - Y_A_R2G2M2)) / np.var(Y_A_R2G2M2)

    ST_A = np.mean(Y_A_R2G2M2**2 - Y_A_R2G2M2 * Y_C1_R2G2M2) / np.var(Y_A_R2G2M2)
    ST_B = np.mean(Y_A_R2G2M2**2 - Y_A_R2G2M2 * Y_C2_R2G2M2) / np.var(Y_A_R2G2M2)
    ST_C = np.mean(Y_A_R2G2M2**2 - Y_A_R2G2M2 * Y_C3_R2G2M2) / np.var(Y_A_R2G2M2)
    
    if print_to_console:
        print('\nParameter sensitivity of system model R2G2M2')
        print('Si_A = %.4f Si_B = %.4f Si_C = %.4f' %(Si_A, Si_B, Si_C))
        print('ST_A = %.4f ST_B = %.4f ST_C = %.4f\n' %(ST_A, ST_B, ST_C))
    
    # ============= Total mean and variance using matrices A ==================
    E_t_d   = np.mean([Y_A_R1G1M1, Y_A_R1G1M2, Y_A_R1G2M1, Y_A_R1G2M2, 
                       Y_A_R2G1M1, Y_A_R2G1M2, Y_A_R2G2M1, Y_A_R2G2M2])
    Var_t_d = np.var([Y_A_R1G1M1, Y_A_R1G1M2, Y_A_R1G2M1, Y_A_R1G2M2, 
                      Y_A_R2G1M1, Y_A_R2G1M2, Y_A_R2G2M1, Y_A_R2G2M2])
    
    # ================================== PSIA =================================
    # First-item 
    E_A2_R1_IT1 = np.mean(Y_B_R1G1M1 * Y_C1_R1G1M1) * (PMB[0] * PMC[0])**2 * PMA[0] + \
                  np.mean(Y_B_R1G1M2 * Y_C1_R1G1M2) * (PMB[0] * PMC[1])**2 * PMA[0] + \
                  np.mean(Y_B_R1G2M1 * Y_C1_R1G2M1) * (PMB[1] * PMC[0])**2 * PMA[0] + \
                  np.mean(Y_B_R1G2M2 * Y_C1_R1G2M2) * (PMB[1] * PMC[1])**2 * PMA[0]
    
    E_A2_R2_IT1 = np.mean(Y_B_R2G1M1 * Y_C1_R2G1M1) * (PMB[0] * PMC[0])**2 * PMA[1] + \
                  np.mean(Y_B_R2G1M2 * Y_C1_R2G1M2) * (PMB[0] * PMC[1])**2 * PMA[1] + \
                  np.mean(Y_B_R2G2M1 * Y_C1_R2G2M1) * (PMB[1] * PMC[0])**2 * PMA[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C1_R2G2M2) * (PMB[1] * PMC[1])**2 * PMA[1]
                  
    E_A2_IT1 = E_A2_R1_IT1 + E_A2_R2_IT1
    
    # Second-item 
    E_A2_R1_IT2 = np.mean(Y_B_R1G1M1 * Y_C1_R1G1M2) * (PMB[0] * PMC[0]) * (PMB[0] * PMC[1]) * PMA[0] + \
                  np.mean(Y_B_R1G1M1 * Y_C1_R1G2M1) * (PMB[0] * PMC[0]) * (PMB[1] * PMC[0]) * PMA[0] + \
                  np.mean(Y_B_R1G1M1 * Y_C1_R1G2M2) * (PMB[0] * PMC[0]) * (PMB[1] * PMC[1]) * PMA[0] + \
                  np.mean(Y_B_R1G1M2 * Y_C1_R1G1M1) * (PMB[0] * PMC[1]) * (PMB[0] * PMC[0]) * PMA[0] + \
                  np.mean(Y_B_R1G1M2 * Y_C1_R1G2M1) * (PMB[0] * PMC[1]) * (PMB[1] * PMC[0]) * PMA[0] + \
                  np.mean(Y_B_R1G1M2 * Y_C1_R1G2M2) * (PMB[0] * PMC[1]) * (PMB[1] * PMC[1]) * PMA[0] + \
                  np.mean(Y_B_R1G2M1 * Y_C1_R1G1M1) * (PMB[1] * PMC[0]) * (PMB[0] * PMC[0]) * PMA[0] + \
                  np.mean(Y_B_R1G2M1 * Y_C1_R1G1M2) * (PMB[1] * PMC[0]) * (PMB[0] * PMC[1]) * PMA[0] + \
                  np.mean(Y_B_R1G2M1 * Y_C1_R1G2M2) * (PMB[1] * PMC[0]) * (PMB[1] * PMC[1]) * PMA[0] + \
                  np.mean(Y_B_R1G2M2 * Y_C1_R1G1M1) * (PMB[1] * PMC[1]) * (PMB[0] * PMC[0]) * PMA[0] + \
                  np.mean(Y_B_R1G2M2 * Y_C1_R1G1M2) * (PMB[1] * PMC[1]) * (PMB[0] * PMC[1]) * PMA[0] + \
                  np.mean(Y_B_R1G2M2 * Y_C1_R1G2M1) * (PMB[1] * PMC[1]) * (PMB[1] * PMC[0]) * PMA[0]
    
    E_A2_R2_IT2 = np.mean(Y_B_R2G1M1 * Y_C1_R2G1M2) * (PMB[0] * PMC[0]) * (PMB[0] * PMC[1]) * PMA[1] + \
                  np.mean(Y_B_R2G1M1 * Y_C1_R2G2M1) * (PMB[0] * PMC[0]) * (PMB[1] * PMC[0]) * PMA[1] + \
                  np.mean(Y_B_R2G1M1 * Y_C1_R2G2M2) * (PMB[0] * PMC[0]) * (PMB[1] * PMC[1]) * PMA[1] + \
                  np.mean(Y_B_R2G1M2 * Y_C1_R2G1M1) * (PMB[0] * PMC[1]) * (PMB[0] * PMC[0]) * PMA[1] + \
                  np.mean(Y_B_R2G1M2 * Y_C1_R2G2M1) * (PMB[0] * PMC[1]) * (PMB[1] * PMC[0]) * PMA[1] + \
                  np.mean(Y_B_R2G1M2 * Y_C1_R2G2M2) * (PMB[0] * PMC[1]) * (PMB[1] * PMC[1]) * PMA[1] + \
                  np.mean(Y_B_R2G2M1 * Y_C1_R2G1M1) * (PMB[1] * PMC[0]) * (PMB[0] * PMC[0]) * PMA[1] + \
                  np.mean(Y_B_R2G2M1 * Y_C1_R2G1M2) * (PMB[1] * PMC[0]) * (PMB[0] * PMC[1]) * PMA[1] + \
                  np.mean(Y_B_R2G2M1 * Y_C1_R2G2M2) * (PMB[1] * PMC[0]) * (PMB[1] * PMC[1]) * PMA[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C1_R2G1M1) * (PMB[1] * PMC[1]) * (PMB[0] * PMC[0]) * PMA[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C1_R2G1M2) * (PMB[1] * PMC[1]) * (PMB[0] * PMC[1]) * PMA[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C1_R2G2M1) * (PMB[1] * PMC[1]) * (PMB[1] * PMC[0]) * PMA[1]
    
    E_A2_IT2 = E_A2_R1_IT2 + E_A2_R2_IT2
    
    SI_A = ((E_A2_IT1 + E_A2_IT2) - E_t_d**2) / (Var_t_d + 1e-20)
    
    # print('SI_A = %.4f\n' %SI_A)
    
    # ================================== PSIB =================================
    # First-item
    E_B2_G1_IT1 = np.mean(Y_B_R1G1M1 * Y_C2_R1G1M1) * (PMA[0] * PMC[0])**2 * PMB[0] + \
                  np.mean(Y_B_R1G1M2 * Y_C2_R1G1M2) * (PMA[0] * PMC[1])**2 * PMB[0] + \
                  np.mean(Y_B_R2G1M1 * Y_C2_R2G1M1) * (PMA[1] * PMC[0])**2 * PMB[0] + \
                  np.mean(Y_B_R2G1M2 * Y_C2_R2G1M2) * (PMA[1] * PMC[1])**2 * PMB[0]
    
    E_B2_G2_IT1 = np.mean(Y_B_R1G2M1 * Y_C2_R1G2M1) * (PMA[0] * PMC[0])**2 * PMB[1] + \
                  np.mean(Y_B_R1G2M2 * Y_C2_R1G2M2) * (PMA[0] * PMC[1])**2 * PMB[1] + \
                  np.mean(Y_B_R2G2M1 * Y_C2_R2G2M1) * (PMA[1] * PMC[0])**2 * PMB[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C2_R2G2M2) * (PMA[1] * PMC[1])**2 * PMB[1]
                  
    E_B2_IT1 = E_B2_G1_IT1 + E_B2_G2_IT1      
                  
    # Second-item      
    E_B2_G1_IT2 = np.mean(Y_B_R1G1M1 * Y_C2_R1G1M2) * (PMA[0] * PMC[0]) * (PMA[0] * PMC[1]) * PMB[0] + \
                  np.mean(Y_B_R1G1M1 * Y_C2_R2G1M1) * (PMA[0] * PMC[0]) * (PMA[1] * PMC[0]) * PMB[0] + \
                  np.mean(Y_B_R1G1M1 * Y_C2_R2G1M2) * (PMA[0] * PMC[0]) * (PMA[1] * PMC[1]) * PMB[0] + \
                  np.mean(Y_B_R1G1M2 * Y_C2_R1G1M1) * (PMA[0] * PMC[1]) * (PMA[0] * PMC[0]) * PMB[0] + \
                  np.mean(Y_B_R1G1M2 * Y_C2_R2G1M1) * (PMA[0] * PMC[1]) * (PMA[1] * PMC[0]) * PMB[0] + \
                  np.mean(Y_B_R1G1M2 * Y_C2_R2G1M2) * (PMA[0] * PMC[1]) * (PMA[1] * PMC[1]) * PMB[0] + \
                  np.mean(Y_B_R2G1M1 * Y_C2_R1G1M1) * (PMA[1] * PMC[0]) * (PMA[0] * PMC[0]) * PMB[0] + \
                  np.mean(Y_B_R2G1M1 * Y_C2_R1G1M2) * (PMA[1] * PMC[0]) * (PMA[0] * PMC[1]) * PMB[0] + \
                  np.mean(Y_B_R2G1M1 * Y_C2_R2G1M2) * (PMA[1] * PMC[0]) * (PMA[1] * PMC[1]) * PMB[0] + \
                  np.mean(Y_B_R2G1M2 * Y_C2_R1G1M1) * (PMA[1] * PMC[1]) * (PMA[0] * PMC[0]) * PMB[0] + \
                  np.mean(Y_B_R2G1M2 * Y_C2_R1G1M2) * (PMA[1] * PMC[1]) * (PMA[0] * PMC[1]) * PMB[0] + \
                  np.mean(Y_B_R2G1M2 * Y_C2_R2G1M1) * (PMA[1] * PMC[1]) * (PMA[1] * PMC[0]) * PMB[0]
    
    E_B2_G2_IT2 = np.mean(Y_B_R1G2M1 * Y_C2_R1G2M2) * (PMA[0] * PMC[0]) * (PMA[0] * PMC[1]) * PMB[1] + \
                  np.mean(Y_B_R1G2M1 * Y_C2_R2G2M1) * (PMA[0] * PMC[0]) * (PMA[1] * PMC[0]) * PMB[1] + \
                  np.mean(Y_B_R1G2M1 * Y_C2_R2G2M2) * (PMA[0] * PMC[0]) * (PMA[1] * PMC[1]) * PMB[1] + \
                  np.mean(Y_B_R1G2M2 * Y_C2_R1G2M1) * (PMA[0] * PMC[1]) * (PMA[0] * PMC[0]) * PMB[1] + \
                  np.mean(Y_B_R1G2M2 * Y_C2_R2G2M1) * (PMA[0] * PMC[1]) * (PMA[1] * PMC[0]) * PMB[1] + \
                  np.mean(Y_B_R1G2M2 * Y_C2_R2G2M2) * (PMA[0] * PMC[1]) * (PMA[1] * PMC[1]) * PMB[1] + \
                  np.mean(Y_B_R2G2M1 * Y_C2_R1G2M1) * (PMA[1] * PMC[0]) * (PMA[0] * PMC[0]) * PMB[1] + \
                  np.mean(Y_B_R2G2M1 * Y_C2_R1G2M2) * (PMA[1] * PMC[0]) * (PMA[0] * PMC[1]) * PMB[1] + \
                  np.mean(Y_B_R2G2M1 * Y_C2_R2G2M2) * (PMA[1] * PMC[0]) * (PMA[1] * PMC[1]) * PMB[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C2_R1G2M1) * (PMA[1] * PMC[1]) * (PMA[0] * PMC[0]) * PMB[1]  + \
                  np.mean(Y_B_R2G2M2 * Y_C2_R1G2M2) * (PMA[1] * PMC[1]) * (PMA[0] * PMC[1]) * PMB[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C2_R2G2M1) * (PMA[1] * PMC[1]) * (PMA[1] * PMC[0]) * PMB[1]
                  
    E_B2_IT2 = E_B2_G1_IT2 + E_B2_G2_IT2     
    
    SI_B = ((E_B2_IT1 + E_B2_IT2) - E_t_d**2) / (Var_t_d + 1e-20)
    
    # print('SI_B = %.4f\n' %SI_B)
    
    # ================================== PSIC =================================
    E_C2_M1_IT1 = np.mean(Y_B_R1G1M1 * Y_C3_R1G1M1) * (PMA[0] * PMB[0])**2 * PMC[0] + \
                  np.mean(Y_B_R1G2M1 * Y_C3_R1G2M1) * (PMA[0] * PMB[1])**2 * PMC[0] + \
                  np.mean(Y_B_R2G1M1 * Y_C3_R2G1M1) * (PMA[1] * PMB[0])**2 * PMC[0] + \
                  np.mean(Y_B_R2G2M1 * Y_C3_R2G2M1) * (PMA[1] * PMB[1])**2 * PMC[0]
    
    E_C2_M2_IT1 = np.mean(Y_B_R1G1M2 * Y_C3_R1G1M2) * (PMA[0] * PMB[0])**2 * PMC[1] + \
                  np.mean(Y_B_R1G2M2 * Y_C3_R1G2M2) * (PMA[0] * PMB[1])**2 * PMC[1] + \
                  np.mean(Y_B_R2G1M2 * Y_C3_R2G1M2) * (PMA[1] * PMB[0])**2 * PMC[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C3_R2G2M2) * (PMA[1] * PMB[1])**2 * PMC[1]
    
    E_C2_IT1 = E_C2_M1_IT1 + E_C2_M2_IT1 
    
    E_C2_M1_IT2 = np.mean(Y_B_R1G1M1 * Y_C3_R1G2M1) * (PMA[0] * PMB[0]) * (PMA[0] * PMB[1]) * PMC[0] + \
                  np.mean(Y_B_R1G1M1 * Y_C3_R2G1M1) * (PMA[0] * PMB[0]) * (PMA[1] * PMB[0]) * PMC[0] + \
                  np.mean(Y_B_R1G1M1 * Y_C3_R2G2M1) * (PMA[0] * PMB[0]) * (PMA[1] * PMB[1]) * PMC[0] + \
                  np.mean(Y_B_R1G2M1 * Y_C3_R1G1M1) * (PMA[0] * PMB[1]) * (PMA[0] * PMB[0]) * PMC[0] + \
                  np.mean(Y_B_R1G2M1 * Y_C3_R2G1M1) * (PMA[0] * PMB[1]) * (PMA[1] * PMB[0]) * PMC[0] + \
                  np.mean(Y_B_R1G2M1 * Y_C3_R2G2M1) * (PMA[0] * PMB[1]) * (PMA[1] * PMB[1]) * PMC[0] + \
                  np.mean(Y_B_R2G1M1 * Y_C3_R1G1M1) * (PMA[1] * PMB[0]) * (PMA[0] * PMB[0]) * PMC[0] + \
                  np.mean(Y_B_R2G1M1 * Y_C3_R1G2M1) * (PMA[1] * PMB[0]) * (PMA[0] * PMB[1]) * PMC[0] + \
                  np.mean(Y_B_R2G1M1 * Y_C3_R2G2M1) * (PMA[1] * PMB[0]) * (PMA[1] * PMB[1]) * PMC[0] + \
                  np.mean(Y_B_R2G2M1 * Y_C3_R1G1M1) * (PMA[1] * PMB[1]) * (PMA[0] * PMB[0]) * PMC[0] + \
                  np.mean(Y_B_R2G2M1 * Y_C3_R1G2M1) * (PMA[1] * PMB[1]) * (PMA[0] * PMB[1]) * PMC[0] + \
                  np.mean(Y_B_R2G2M1 * Y_C3_R2G1M1) * (PMA[1] * PMB[1]) * (PMA[1] * PMB[0]) * PMC[0]
    
    E_C2_M2_IT2 = np.mean(Y_B_R1G1M2 * Y_C3_R1G2M2) * (PMA[0] * PMB[0]) * (PMA[0] * PMB[1]) * PMC[1] + \
                  np.mean(Y_B_R1G1M2 * Y_C3_R2G1M2) * (PMA[0] * PMB[0]) * (PMA[1] * PMB[0]) * PMC[1] + \
                  np.mean(Y_B_R1G1M2 * Y_C3_R2G2M2) * (PMA[0] * PMB[0]) * (PMA[1] * PMB[1]) * PMC[1] + \
                  np.mean(Y_B_R1G2M2 * Y_C3_R1G1M2) * (PMA[0] * PMB[1]) * (PMA[0] * PMB[0]) * PMC[1] + \
                  np.mean(Y_B_R1G2M2 * Y_C3_R2G1M2) * (PMA[0] * PMB[1]) * (PMA[1] * PMB[0]) * PMC[1] + \
                  np.mean(Y_B_R1G2M2 * Y_C3_R2G2M2) * (PMA[0] * PMB[1]) * (PMA[1] * PMB[1]) * PMC[1] + \
                  np.mean(Y_B_R2G1M2 * Y_C3_R1G1M2) * (PMA[1] * PMB[0]) * (PMA[0] * PMB[0]) * PMC[1] + \
                  np.mean(Y_B_R2G1M2 * Y_C3_R1G2M2) * (PMA[1] * PMB[0]) * (PMA[0] * PMB[1]) * PMC[1] + \
                  np.mean(Y_B_R2G1M2 * Y_C3_R2G2M2) * (PMA[1] * PMB[0]) * (PMA[1] * PMB[1]) * PMC[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C3_R1G1M2) * (PMA[1] * PMB[1]) * (PMA[0] * PMB[0]) * PMC[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C3_R1G2M2) * (PMA[1] * PMB[1]) * (PMA[0] * PMB[1]) * PMC[1] + \
                  np.mean(Y_B_R2G2M2 * Y_C3_R2G1M2) * (PMA[1] * PMB[1]) * (PMA[1] * PMB[0]) * PMC[1]
    
    E_C2_IT2 = E_C2_M1_IT2 + E_C2_M2_IT2
    
    SI_C = ((E_C2_IT1 + E_C2_IT2) - E_t_d**2) / (Var_t_d + 1e-20)
    
    # print('SI_C = %.4f\n' %SI_C)
    
    # ================================== PSTA =================================
    # First-item 
    E_A2_G1M1_IT1 = np.mean(Y_A_R1G1M1 * Y_C1_R1G1M1) * PMA[0]**2 * (PMB[0] * PMC[0]) + \
                    np.mean(Y_A_R2G1M1 * Y_C1_R2G1M1) * PMA[1]**2 * (PMB[0] * PMC[0])
                    
    E_A2_G1M2_IT1 = np.mean(Y_A_R1G1M2 * Y_C1_R1G1M2) * PMA[0]**2 * (PMB[0] * PMC[1]) + \
                    np.mean(Y_A_R2G1M2 * Y_C1_R2G1M2) * PMA[1]**2 * (PMB[0] * PMC[1]) 
                    
    E_A2_G2M1_IT1 = np.mean(Y_A_R1G2M1 * Y_C1_R1G2M1) * PMA[0]**2 * (PMB[1] * PMC[0]) + \
                    np.mean(Y_A_R2G2M1 * Y_C1_R2G2M1) * PMA[1]**2 * (PMB[1] * PMC[0])
                    
    E_A2_G2M2_IT1 = np.mean(Y_A_R1G2M2 * Y_C1_R1G2M2) * PMA[0]**2 * (PMB[1] * PMC[1]) + \
                    np.mean(Y_A_R2G2M2 * Y_C1_R2G2M2) * PMA[1]**2 * (PMB[1] * PMC[1])
                    
    IT1 = E_A2_G1M1_IT1 + E_A2_G1M2_IT1 + E_A2_G2M1_IT1 + E_A2_G2M2_IT1 
                    
    # Second-item
    E_A2_G1M1_IT2 = np.mean(Y_A_R1G1M1 * Y_C1_R2G1M1) * PMA[0] * PMA[1] * (PMB[0] * PMC[0]) + \
                    np.mean(Y_A_R2G1M1 * Y_C1_R1G1M1) * PMA[1] * PMA[0] * (PMB[0] * PMC[0])
    
    E_A2_G1M2_IT2 = np.mean(Y_A_R1G1M2 * Y_C1_R2G1M2) * PMA[0] * PMA[1] * (PMB[0] * PMC[1]) + \
                    np.mean(Y_A_R2G1M2 * Y_C1_R1G1M2) * PMA[1] * PMA[0] * (PMB[0] * PMC[1]) 
                 
    E_A2_G2M1_IT2 = np.mean(Y_A_R1G2M1 * Y_C1_R2G2M1) * PMA[0] * PMA[1] * (PMB[1] * PMC[0]) + \
                    np.mean(Y_A_R2G2M1 * Y_C1_R1G2M1) * PMA[1] * PMA[0] * (PMB[1] * PMC[0]) 
    
    E_A2_G2M2_IT2 = np.mean(Y_A_R1G2M2 * Y_C1_R2G2M2) * PMA[0] * PMA[1] * (PMB[1] * PMC[1]) + \
                    np.mean(Y_A_R2G2M2 * Y_C1_R1G2M2) * PMA[1] * PMA[0] * (PMB[1] * PMC[1])
                    
    IT2 = E_A2_G1M1_IT2 + E_A2_G1M2_IT2 + E_A2_G2M1_IT2 + E_A2_G2M2_IT2

    ST_A = 1- ((IT1 + IT2) - E_t_d**2) / (Var_t_d + 1e-20)
    
    # print('ST_A = %.4f\n' %ST_A)
    
    # ================================== PSTB =================================
    # First-item 
    E_B2_R1M1_IT1 = np.mean(Y_A_R1G1M1 * Y_C2_R1G1M1) * PMB[0]**2 * (PMA[0] * PMC[0]) + \
                    np.mean(Y_A_R1G2M1 * Y_C2_R1G2M1) * PMB[1]**2 * (PMA[0] * PMC[0])
                    
    E_B2_R1M2_IT1 = np.mean(Y_A_R1G1M2 * Y_C2_R1G1M2) * PMB[0]**2 * (PMA[0] * PMC[1]) + \
                    np.mean(Y_A_R1G2M2 * Y_C2_R1G2M2) * PMB[1]**2 * (PMA[0] * PMC[1])
                    
    E_B2_R2M1_IT1 = np.mean(Y_A_R2G1M1 * Y_C2_R2G1M1) * PMB[0]**2 * (PMA[1] * PMC[0]) + \
                    np.mean(Y_A_R2G2M1 * Y_C2_R2G2M1) * PMB[1]**2 * (PMA[1] * PMC[0])
                    
    E_B2_R2M2_IT1 = np.mean(Y_A_R2G1M2 * Y_C2_R2G1M2) * PMB[0]**2 * (PMA[1] * PMC[1]) + \
                    np.mean(Y_A_R2G2M2 * Y_C2_R2G2M2) * PMB[1]**2 * (PMA[1] * PMC[1])
                    
    IT1 = E_B2_R1M1_IT1 + E_B2_R1M2_IT1 + E_B2_R2M1_IT1 + E_B2_R2M2_IT1
    
    # Second-item
    E_B2_R1M1_IT2 = np.mean(Y_A_R1G1M1 * Y_C2_R1G2M1) * PMB[0] * PMB[1] * (PMA[0] * PMC[0]) + \
                    np.mean(Y_A_R1G2M1 * Y_C2_R1G1M1) * PMB[1] * PMB[0] * (PMA[0] * PMC[0]) 
    
    E_B2_R1M2_IT2 = np.mean(Y_A_R1G1M2 * Y_C2_R1G2M2) * PMB[0] * PMB[1] * (PMA[0] * PMC[1]) + \
                    np.mean(Y_A_R1G2M2 * Y_C2_R1G1M2) * PMB[1] * PMB[0] * (PMA[0] * PMC[1])
                 
    E_B2_R2M1_IT2 = np.mean(Y_A_R2G1M1 * Y_C2_R2G2M1) * PMB[0] * PMB[1] * (PMA[1] * PMC[0]) + \
                    np.mean(Y_A_R2G2M1 * Y_C2_R2G1M1) * PMB[1] * PMB[0] * (PMA[1] * PMC[0]) 
    
    E_B2_R2M2_IT2 = np.mean(Y_A_R2G1M2 * Y_C2_R2G2M2) * PMB[0] * PMB[1] * (PMA[1] * PMC[1]) + \
                    np.mean(Y_A_R2G2M2 * Y_C2_R2G1M2) * PMB[1] * PMB[0] * (PMA[1] * PMC[1])
	
    IT2 = E_B2_R1M1_IT2 + E_B2_R1M2_IT2 + E_B2_R2M1_IT2 + E_B2_R2M2_IT2

    ST_B = 1- ((IT1 + IT2) - E_t_d**2) / (Var_t_d + 1e-20)
    
    # print('ST_B = %.4f\n' %ST_B)
    
    # ================================== PSTC =================================
    E_C2_R1G1_IT1 = np.mean(Y_A_R1G1M1 * Y_C3_R1G1M1) * PMC[0]**2 * (PMA[0] * PMB[0]) + \
                    np.mean(Y_A_R1G1M2 * Y_C3_R1G1M2) * PMC[1]**2 * (PMA[0] * PMB[0]) 
                    
    E_C2_R1G2_IT1 = np.mean(Y_A_R1G2M1 * Y_C3_R1G2M1) * PMC[0]**2 * (PMA[0] * PMB[1])  + \
                    np.mean(Y_A_R1G2M2 * Y_C3_R1G2M2) * PMC[1]**2 * (PMA[0] * PMB[1]) 
                    
    E_C2_R2G1_IT1 = np.mean(Y_A_R2G1M1 * Y_C3_R2G1M1) * PMC[0]**2 * (PMA[1] * PMB[0])  + \
                    np.mean(Y_A_R2G1M2 * Y_C3_R2G1M2) * PMC[1]**2 * (PMA[1] * PMB[0]) 
                    
    E_C2_R2G2_IT1 = np.mean(Y_A_R2G2M1 * Y_C3_R2G2M1) * PMC[0]**2 * (PMA[1] * PMB[1])  + \
                    np.mean(Y_A_R2G2M2 * Y_C3_R2G2M2) * PMC[1]**2 * (PMA[1] * PMB[1]) 
	
    IT1 = E_C2_R1G1_IT1 + E_C2_R1G2_IT1 + E_C2_R2G1_IT1 + E_C2_R2G2_IT1 
    
    E_C2_R1G1_IT2 = np.mean(Y_A_R1G1M1 * Y_C3_R1G1M2) * PMC[0] * PMB[1] * (PMA[0] * PMB[0]) + \
                    np.mean(Y_A_R1G1M2 * Y_C3_R1G1M1) * PMC[1] * PMB[0] * (PMA[0] * PMB[0])
    
    E_C2_R1G2_IT2 = np.mean(Y_A_R1G2M1 * Y_C3_R1G2M2) * PMC[0] * PMB[1] * (PMA[0] * PMB[1]) + \
                    np.mean(Y_A_R1G2M2 * Y_C3_R1G2M1) * PMC[1] * PMB[0] * (PMA[0] * PMB[1])
                 
    E_C2_R2G1_IT2 = np.mean(Y_A_R2G1M1 * Y_C3_R2G1M2) * PMC[0] * PMB[1] * (PMA[1] * PMB[0]) + \
                    np.mean(Y_A_R2G1M2 * Y_C3_R2G1M1) * PMC[1] * PMB[0] * (PMA[1] * PMB[0]) 
    
    E_C2_R2G2_IT2 = np.mean(Y_A_R2G2M1 * Y_C3_R2G2M2) * PMC[0] * PMB[1] * (PMA[1] * PMB[1]) + \
                    np.mean(Y_A_R2G2M2 * Y_C3_R2G2M1) * PMC[1] * PMB[0] * (PMA[1] * PMB[1])
	
    IT2 = E_C2_R1G1_IT2 + E_C2_R1G2_IT2 + E_C2_R2G1_IT2 + E_C2_R2G2_IT2
    ST_C = 1- ((IT1 + IT2) - E_t_d**2) / (Var_t_d + 1e-20)
    
    # print('ST_C = %.4f\n' %ST_C)
    
   
    return SI_A, SI_B, SI_C, ST_A, ST_B, ST_C

#%% Generate parameter values
N = 64
negatives = True              # All values should be positive       
while negatives:
    pvalues_A, pvalues_B = generate_param_values(N)
    if pvalues_A.min() > 0 and pvalues_A.min() > 0:
        negatives = False
    else:
        # Replace all negative values with 1e-20
        pvalues_A, pvalues_B = generate_param_values(N)
        pvalues_A[pvalues_A < 0] =  1e-20
        pvalues_B[pvalues_B < 0] = 1e-20
    
#%% Compute the sensitivity indices
ret = np.zeros((nx, 6))
for ix, x in enumerate(xs): 
    SI_A, SI_B, SI_C, ST_A, ST_B, ST_C = fast_MC(x, pvalues_A, pvalues_B, print_to_console=False)
    ret[ix, :] = SI_A, SI_B, SI_C, ST_A, ST_B, ST_C 
    print('Computed sensitivity indices for ix=%d' %ix)
    print('    SI_A=%.4f, SI_B=%.4f, SI_C=%.4f\n    ST_A=%.4f, ST_B=%.4f, ST_C=%.4f' %(SI_A, SI_B, SI_C, ST_A, ST_B, ST_C))
    
    
    
plt.figure(figsize=(10, 5))

plt.plot(xs, ret[:, 0] * 100, label='$PS_K$' + ' for R', color='r', linestyle='-', lw=2, alpha=0.8)
plt.plot(xs, ret[:, 1] * 100, label='$PS_K$' + ' for G', color='b', linestyle='-',  lw=2, alpha=0.8)
plt.plot(xs, ret[:, 2] * 100, label='$PS_K$' + ' for M', color='g', linestyle='-',  lw=2, alpha=0.8)

plt.plot(xs, ret[:, 3] * 100, label='$PS_{TK}$' + ' for R', color='r', linestyle='-.',  lw=2, alpha=1)
plt.plot(xs, ret[:, 4] * 100, label='$PS_{TK}$' + ' for G', color='b', linestyle='-.',  lw=2, alpha=0.8)
plt.plot(xs, ret[:, 5] * 100, label='$PS_{TK}$' + ' for M', color='g', linestyle='-.',  lw=2, alpha=0.8)

plt.xlabel('x [m]', fontsize=14)
plt.ylabel('Process sensitivity (%)',  fontsize=14)
plt.xlim([0, L])
plt.ylim([0, 100])
plt.legend(frameon=False, loc='center')

plt.show()





