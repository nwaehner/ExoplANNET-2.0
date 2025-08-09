from tqdm import tqdm
import numpy as np
import random
import argparse
import os
import gzip
import functions as func
import json
import math as ma
import pandas as pd
from astropy.timeseries import LombScargle


''''
Use with the following command. You should specify the number of stars (int: STARS) and the amount of planets on each star (int: PLANETS)

"python ./rvsimu.py -stars STARS -pl PLANETS outputDir"


positional arguments:
  STARS                   Amount of synthetic RV samples
  PLANETS                 Amount of planets on each star
  outputDir               Output Dir

optional arguments:
  -h, --help            show this help message and exit
  -n NOISE, --noise NOISE
                        NN=No Noise,WN=White Noise, CN=Correlated Noise (Default = CN)
  -a AMPLITUDE, --amplitude AMPLITUDE 
                        Minimum amplitude of the generated planets (Default = 0.1)
  -p PERIOD, --period PERIOD
                        Minimum period of the generated planets (Default = 5.0)
  -s SEED, --seed SEED  
                        Seed used to generate the samples (Default = random(0, 2**32 - 1))

'''


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


version='6.0'

parser = argparse.ArgumentParser(description="Synthetic radial velocity generator.")
parser.add_argument("-stars",help="Amount of synthetic RV samples", type=int)
parser.add_argument("outputDir",help="Output Dir", type=str)
parser.add_argument("-n","--noise", help="NN=No Noise,WN=White Noise, CN=Correlated Noise(Default)",type=str, default='CN')
parser.add_argument("-pl","--planets", help="Number of planets",type=int)
parser.add_argument("-a","--amplitude", help="Minimum amplitude of the generated planets",type=float, default=0.1)
parser.add_argument("-p","--period", help="Minimum period of the generated planets",type=float, default=5.0) 
parser.add_argument("-seed","--seed",help="Seed used to generate the samples", type=int, default=random.randrange(0, 2**32 - 1))
results = parser.parse_args()

SEED = results.seed
np.random.seed(SEED)
random.seed(SEED)

#We cannot use default parameter because we want to regenerate everything with the seed
if(results.planets == None):
    PLANETS = random.randrange(15)
else:
    PLANETS=results.planets

MIN_PL_AMP=results.amplitude
MIN_PL_PERIOD=results.period

if(results.noise.upper() not in ['NN', 'WN', 'CN']):
    NOISETYPE = 'CN'
else:
    NOISETYPE = results.noise.upper()
    
# STARS, the total amount of stars
# PLANETS, how many planets
# MIN_PL_AMP, minimum planets amplitude
# MIN_PL_PERIOD, minimum planets period

STARS=results.stars

outDir=results.outputDir  
outCat = "/"+str(f"{STARS} Stars with {PLANETS} planets")
fileName = "/"+str(STARS)+"-"+NOISETYPE + "-P"+str(PLANETS)+"-AMP"+str(MIN_PL_AMP)+"-PER"+str(MIN_PL_PERIOD)+"-SEED"+str(SEED)+"-V"+str(version) +".json.gz"
filename = outDir+outCat+fileName


print("\n=------------------------------------")
print(" Synthetic Radial Velocity Generator")
print("             Version:", version)
print("=------------------------------------ \n ")
print("Generating:",STARS, "Star/s", "sampled using a random time distribution of stars measured by HARPS")
print("Noise:",NOISETYPE)
print("Planets:", PLANETS)
print("Minimum Planets Amplitude:", MIN_PL_AMP)
print("Minimum Planets Period:", MIN_PL_PERIOD)
print("Seed:", SEED)
print("Output:", filename)

os.makedirs(os.path.dirname(filename), exist_ok=True)

metadataDicc = {
		"numStars" : STARS,
		"noise" : NOISETYPE,
        "numPlanets" : PLANETS,    
		"minPlAmp" : MIN_PL_AMP,
		"minPlPeriod" : MIN_PL_PERIOD,
		"seed" : SEED,
        "version" : version
	}


#Open all observation calendars

Harps = pd.read_pickle(r"C:\Users\Administrator\Desktop\facultad\Tesis\Codigo\Simulations_github\Harps_calendar_observations.pkl")
Harps = Harps.drop(columns=["index"])

tiempos = Harps["obj_date_mins"].values/(60*24) # From minutes to days
estrellas = Harps["obj_id_catname"].values
catalogos = Harps["prog_id"].values

# Search for max_t in order to define the optimal frequency array for sampling all periodograms.
max_t = 0
for i in range(len(tiempos)):
    if max(tiempos[i]) > max_t:
        max_t = max(tiempos[i])

frec_muestreo = 1/(2*(max_t))

with gzip.open(filename, 'wb') as file:
    # get stars from Udry
    res, cond = func.get_stars_udry() 
    stars=[metadataDicc]     
    
    for i in tqdm(range(0, STARS), desc="Processing stars", unit="star"):
        
        # Pick random observation calendar
        random_number = random.randint(0, len(tiempos)-1)
        t = tiempos[random_number] 
        star_name = estrellas[random_number]

        # Randomize 10% of the array
        t = func.augmentation(t)
        
        # Pick mean error randomly in m/s
        meanerror = np.random.choice(res.loc[cond, 'msvrad']) * 1e3
        DAYS = len(t)

        err = np.random.randn(DAYS) * 0.3  + meanerror
        # Set floor to minerror/2    
        min_error=meanerror/2.0
        replaceerror=meanerror/2.0
        err = np.where(err < min_error, replaceerror, err)

        rv_planets, planets = func.generateN_PL(DAYS, PLANETS, t, MIN_PL_AMP, MIN_PL_PERIOD)

        # RV without planets 
        rv_wop, nu, psd, star_rot = func.generate_RV_v5(DAYS,t, NOISETYPE, res, cond, err)

        # Total RV
        rv_with_pl = rv_wop + rv_planets
        
        # Frequency array 
        pnu = np.arange(1/250, 1/2, frec_muestreo)
        prange = 1.0/pnu

        periodogram = LombScargle(t, rv_with_pl, err).power(pnu)
        
        rv_dict = {
                "rv_planets" : rv_planets,                             
                "rv_wo_pl" : rv_wop,             
                "rv_star" : rv_with_pl,             
        }

        star_dict = {
                "star_rot" : star_rot,
                "prange" : prange, 
                "t" : t, 
                "star_name": star_name,
                "err" : err,
                "planets" : planets,
                "periodogram" : periodogram,
                "rv": rv_dict
                }

        stars.append(star_dict)

    allStars = json.dumps(stars, cls=NumpyEncoder,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)

    file.write(allStars.encode())
    
    
print("\n...Output Generated Successfully!!!")      