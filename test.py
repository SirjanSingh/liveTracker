from skyfield.api import load
from skyfield.data import hipparcos
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load Hipparcos star catalog
# filename = load.download(hipparcos.URL)
with load.open('star_data/hip_main.dat') as f:
    stars = hipparcos.load_dataframe(f)
# Load Hipparcos catalog properly
# stars = load.download('https://astronomyapi.nyc3.digitaloceanspaces.com/hipparcos.dat')
pd.set_option('display.max_columns', 50)

print(stars)
