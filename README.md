# ExoplANNET-2.0
A convolutional neural network to detect planetary signals on radial velocity time series

<p align="center">
  <img src="https://github.com/user-attachments/assets/009cb1a4-529a-49ec-ac5d-8fa65efe3c9e" width="100%" alt="Demo Image">
</p>

# Abstract

As my thesis project, I built upon [previous work](https://www.aanda.org/articles/aa/full_html/2023/09/aa46417-23/aa46417-23.html) by Luis A. Nieto and Rodrigo F. Díaz.

In the thesis, we simulated radial velocity (RV) time series on multiple real observation calendars, typical in the search for exoplanets. The RV time series included both stellar activity and planetary signals. For each one, we computed the Generalized Lomb-Scargle (GLS) periodogram and trained a neural network (NN) to determine whether the maximum peak was due to the presence of a planetary companion.

Comparing our results with the false alarm probability (FAP) method, we obtained 52% fewer false positives without increasing the number of false negatives. We also tested the method on real data, obtaining good performance. These results suggest that the NN approach may outperform the statistical methods currently in use. For further details, my thesis (in Spanish) describes the full work and methodology used.

## Upgrades made with respect to the previous work:
- Generalized the observation calendars to apply the neural network to real time series.
- Trained with an extended range of planetary periods.
- Calculated the FAP exclusively using the Monte Carlo method (for robustness and reliability).
- Added an attention layer to improve interpretability, which also resulted in better overall performance.


# Code content

This repository contains the code used for simulation (based on the previously done by L. A. Nieto for his paper), the trained neural network and some examples for testing. 

Below are examples showing the simulated RV data (sampled on real stellar observation calendars), the GLS periodogram, and the NN prediction.

<p align="center">
  <img src="https://github.com/user-attachments/assets/90ccbf27-cf42-4e8e-98bf-ac308d15e375" alt="imagen_buena" width="80%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/eb802500-8d2f-41af-bbd7-f7bb42b1e9cf" alt="imagen_buena_2" width="80%">
</p>

# Use
To make it work, replace the paths at the beginning of functions.py. Then, use the following command to run the simulation, specifying the number of stars (int: STARS) and the number of planets per star (int: PLANETS):

```
"python ./rvsimu.py -stars STARS -pl PLANETS outputDir"
```

Positional arguments:
- **STARS**: Amount of synthetic RV samples  
- **PLANETS**: Amount of planets on each star  
- **outputDir**: Output directory  

Optional arguments:
- `-h`, `--help`: Show this help message and exit  
- `-n NOISE`, `--noise NOISE`: NN=No Noise, WN=White Noise, CN=Correlated Noise (Default = CN)  
- `-a AMPLITUDE`, `--amplitude AMPLITUDE`: Minimum amplitude of the generated planets (Default = 0.1)  
- `-p PERIOD`, `--period PERIOD`: Minimum period of the generated planets (Default = 5.0)  
- `-s SEED`, `--seed SEED`: Seed used to generate the samples (Default = random(0, 2**32 - 1)) 

# Contact

For any ideas or inquiries, contact me at: nicolaswaehner@gmail.com
