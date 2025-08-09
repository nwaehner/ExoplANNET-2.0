"""
Created on Wed Jul 18 16:27:23 2018

@author: agus
"""

import numpy as np

##### Beta Hydri values
Atc = np.array([0.055, 0.021,11.6e-3])
Btc = np.array([24.3*3600, 3.4*3600, 72.8*60]) # in Secs
Ctc = np.array([4.3, 4.0, 5.0])
AlTc = 18.5e-3
GammaTc = 0.17*1000 # in microheartz
v0 = 1.0 * 1000 # in microheartz
ConstTc = 2.6e-4

BetaHydri = (Atc, Btc, Ctc, AlTc, GammaTc, v0, ConstTc)

##### Mu Ara values
Atc = np.array([0.029, 0.027,1.1e-3])
Btc = np.array([13*3600, 3.4*3600, 43.8*60]) # in Secs
Ctc = np.array([6.0, 5.0, 4.5])
AlTc = 3.2e-3
GammaTc = 0.26*1000 # in microheartz
v0 = 1.9 * 1000 # in microheartz
ConstTc = 5.1e-4

MuAra = (Atc, Btc, Ctc, AlTc, GammaTc, v0, ConstTc)

##### Alpha Cen A values
Atc = np.array([0.027, 0.003,0.3e-3])
Btc = np.array([7.4*3600, 1.2*3600, 17.9*60]) # in Secs
Ctc = np.array([3.1, 3.9, 8.9])
AlTc = 2.6e-3
GammaTc = 0.36*1000 # in microheartz
v0 = 2.4 * 1000 # in microheartz
ConstTc = 1.4e-4

AlphaCenA = (Atc, Btc, Ctc, AlTc, GammaTc, v0, ConstTc)

##### Tau Ceti values
Atc = np.array([0.027, 0.002,0.3e-3])
Btc = np.array([6.7*3600, 1.2*3600, 18.5*60]) # in Secs
Ctc = np.array([2.6, 8.9, 19.8])
GammaTc = 0.75*1000 # in microheartz
AlTc = 0.3e-3
v0 = 4.5 * 1000 # in microheartz
ConstTc = 1.7e-4

TauCeti = (Atc, Btc, Ctc, AlTc, GammaTc, v0, ConstTc)

##### Alpha Cen B values
Atc = np.array([0.002, 0.001,0.1e-3])
Btc = np.array([12*3600, 0.7*3600, 8.9*60]) # in Secs
Ctc = np.array([4.8, 4.4, 7.5])
GammaTc = 0.68*1000 # in microheartz
AlTc = 0.2e-3
v0 = 3.9 * 1000 # in microheartz
ConstTc = 0.5e-4

AlphaCenB = (Atc, Btc, Ctc, AlTc, GammaTc, v0, ConstTc)