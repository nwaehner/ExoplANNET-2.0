import numpy as np
import stars_const as stars
import pandas as pd
import os
import random
import copy
import argparse
import scipy.stats as st
from numba import njit
import sys



##############################

# IMPORTANT:

# add path to tables and pypgr
tabledir = r'...\tables'
sys.path.insert(1, r'...\pygpr\pygpr')

import kernels
import core
import gaussianprocess


##############################


def read_catalog(catalogfile=None):
    if catalogfile is None:
        catalogfile = os.path.join(tabledir, 'HARPS_Udry_LP.cat')
    # Read LP catalogue
    catlp = pd.read_table(catalogfile, header=0, skiprows=[1, ], index_col=0)

    return catlp

def read_resum(resumfile=None):
    if resumfile is None:
        resumfile = os.path.join(tabledir, 'resum_harps.rdb')

    # Read observations from resum file
    res = pd.read_table(resumfile, header=0, skiprows=[1, ],
                        index_col=['name'])

    res.loc[:, 'variability'] = pd.Series(res['wrms'] / res['msvrad'],
                                          index=res.index)

    return res

def augmentation(t):
    """
    Augments an array by randomly deleting or changing up to 10% of its elements.
    
    Parameters:
    - t (list or np.ndarray): The input array to augment.
    
    Returns:
    - np.ndarray: The augmented array.
    """
    t = np.array(t)
    n = len(t)
    max_modifications = int(n * 0.1) 
    
    # decide how many elements to delete or change
    num_modifications = np.random.randint(0, max_modifications + 1)
    # select indices for modification
    indices_to_modify = np.random.choice(n, num_modifications, replace=False)
    
    augmented_t = np.copy(t)
    
    # Perform modifications
    for idx in indices_to_modify:
        if np.random.rand() < 0.5:  # 50% chance to delete
            augmented_t[idx] = -1  # delete it
        else:  # 50% chance to change
            augmented_t[idx] = np.random.uniform(min(t), max(t))  # Replace with a random value
    
    augmented_t = augmented_t[augmented_t != -1]  # Filter out -1 values
    return augmented_t

###### Parser stuff
def check_gt_one(value):
    ivalue = int(value)
    if ivalue < 1:
         raise argparse.ArgumentTypeError("%s The value must be greater than 1" % value)
    return ivalue
###
    
def get_stars_udry():  
    # Read summary file
    res = read_resum()
    # Read Udry Large Programme catalogue
    cat = read_catalog()    
    # Select only stars in the Udry catalogue with more than 40 observations.
    cond = res.index.isin(cat.index) * res['nmes'] > 40
    
    return res, cond

       
##### Some Functions
def lorentz(Vi, V0, Al, Gamma):
    return Al*  (Gamma**2/((Vi-V0)**2+Gamma**2))

def ptot(vi, A, B, C):
    B_Ms = B/1e6 # B in megaseconds = 1e6 sec.
    return (A/(((vi*B_Ms)**C)+1)).sum()


#### Generate stellar activity signals


def generate_RV_v5(DAYS, t, Noise, res, cond, err):
    
    # Velocities vector (v)
    # this is the main core, we want to generate the most realistics values, so we add some stellar noise
    # - Pulsations
    # - Granulation
    # - Rotational modulation
    
    #first, we add some white noise using the error vector
    rv_wn = np.random.randn(len(err)) * err

    # for the first two elements we use the procedures detailed in
    # http://esoads.eso.org/abs/2011A%26A...525A.140D
    
    # Define frequency nu array
    T = np.max(t) - np.min(t)
    paso_inv_dias = 100/T
    nu = np.arange(1/T, 100, paso_inv_dias)
    
    # Each nu = 1/Days => 1/(86400 s) => 1/86400 Hz => 1.157e-5 hz => 11.57 microHz 
    

    #nu = nu/(3600*24) * 10**(6)  # 10**6/(3600*24) = 11.57. Estaba bien lo de abajo.
    nu = nu * 11.57 #En microHz
    paso_microhz = paso_inv_dias*11.57 
    
    #ramdomly select star parameters
    Params=[stars.BetaHydri,stars.MuAra,stars.AlphaCenA,stars.TauCeti,stars.AlphaCenB]

    Atc,Btc,Ctc,AlTc,GammaTc,v0,ConstTc = random.choice(Params)
    
    ResPtot=np.array([ptot(vi, Atc, Btc, Ctc) for vi in nu])
    ResLorentz=np.array([lorentz(vi, v0, AlTc, GammaTc) for vi in nu])
    
    #psd = COMPUTE_PSD(nu)
    psd = ResPtot + ResLorentz + ConstTc
    
    ## Choose phase randomly [0, 2*pi]
    phase = np.random.rand(len(nu)) * 2*np.pi
    
    
    rv_gran = np.sum(np.sqrt(psd*paso_microhz) * np.sin(2*np.pi*nu*t[None,:].T + phase), axis=1) #ARREGLADO. IBA PASO_MICROHZ
    
    ############# Rotational Modulation.
    # Choose hyperparameters randomly
    P = np.random.choice(res.loc[cond * res['prot'] > 0, 'prot'])
    epsilon = np.random.rand()*0.5 + 0.5
    tau = np.random.randn() * 0.1*P + 3*P
    A = st.gamma.rvs(2.0, scale=0.5)
    #print("Rotación:", P)

    #V5, guardamos todos los parámetros de las estrellas base
    host_star = dict()
    host_star['P']=P

    ## quasi-periodic kernel with drawn hyperparameters
    qpk = kernels.QuasiPeriodicKernel([A, tau, P, epsilon])
    
    # Gaussian process with this kernel and evaluated at the times of the simulation.
    gp = gaussianprocess.GaussianProcess(qpk, t)
    
    # Now sample from that Gaussian Process as many curves as we want.
    rv_rot = gp.sample()[0]

    #TODO, hacerlo antes
    if(Noise =='NN'):   
        rv_wop = np.zeros(DAYS)
    elif(Noise =='WN'):
        rv_wop = rv_wn
    else: #CN
        rv_wop = rv_wn + rv_gran + rv_rot
        
    return rv_wop, nu, psd, P


########## Generate planets


def generateN_PL(DAYS, PLANETS, t,minPlAmp = 0.1, minPlPeriod = 5):
    
    ### generamos planetas con una frecuencia de separacion de 3/200 
    #P_pl períodos de planetas        
    maxPlPeriod = 200
    DeltaT = DAYS
    P_pls = []
    P_pls.append( np.random.uniform(minPlPeriod,maxPlPeriod) ) #Agrego el primer planeta
    
    for i in range(1,PLANETS):
        P_i = np.nan
        while np.isnan(P_i):
            P_cand = np.random.uniform(minPlPeriod,maxPlPeriod)    
            # Veo que los períodos se separen al menos en un factor 1.1 del resto
            overlap = [(P_cand/P_j > 1.1 or P_j/P_cand > 1.1) for P_j in P_pls]
            
            if np.array(overlap).all() == True: #Si todos difieren mas de 1.1 lo agrego
                P_i = P_cand

        P_pls.append(P_i)
        
    T0_pls = np.random.randn(PLANETS) * DAYS #normal   
    #amplitudes
    xmin = minPlAmp
    xmax = 10
    q = np.random.rand(PLANETS)  #uniform
    K_pls= xmin * (xmax / xmin)**q
    

    #P_pls -> Planets Periods
    #K_pls -> planets Amplitudes
    #T0_pls -> Planets T0

    #build the planet list and the planets rv
    resPlanets=[]
    rv_pls = np.zeros(DAYS)
    for i in range(0, PLANETS):
        
        planetDicc = {
                "P" : P_pls[i],
                "K" : K_pls[i],
                "T0" : T0_pls[i]
            }
            
        resPlanets.append(planetDicc)
    
        
        rv_pls += K_pls[i]* np.sin( 2 * np.pi / P_pls[i] * (t - T0_pls[i]) )
        

    return rv_pls, resPlanets
