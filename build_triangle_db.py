# build_triangle_db.py
import pickle
import numpy as np
import math
from itertools import combinations
from skyfield.api import load
from skyfield.data import hipparcos
from scipy.spatial import KDTree

# Config
CATALOG_FILE = 'star_data/hip_main.dat'
BRIGHT_MAG   = 5.0

# 1) Load & filter catalog
with load.open(CATALOG_FILE) as f:
    stars = hipparcos.load_dataframe(f)
stars = stars[stars['magnitude'] < BRIGHT_MAG].reset_index(drop=True)

# helper to convert to 3D unit vector
def to_vec(ra, dec):
    rad_ra, rad_dec = math.radians(ra), math.radians(dec)
    return np.array([
        math.cos(rad_dec)*math.cos(rad_ra),
        math.cos(rad_dec)*math.sin(rad_ra),
        math.sin(rad_dec),
    ])

star_vecs = np.stack([to_vec(ra, dec)
                      for ra, dec in zip(stars['ra_degrees'], stars['dec_degrees'])])

# 2) Build triangle keys & values
keys, vals = [], []
for i,j,k in combinations(range(len(star_vecs)), 3):
    # angular distances
    d = sorted([ 
        np.arccos(np.clip(np.dot(star_vecs[i], star_vecs[j]), -1,1)),
        np.arccos(np.clip(np.dot(star_vecs[j], star_vecs[k]), -1,1)),
        np.arccos(np.clip(np.dot(star_vecs[k], star_vecs[i]), -1,1))
    ])
    if d[2] == 0: 
        continue
    keys.append((d[0]/d[2], d[1]/d[2]))
    vals.append((i, j, k))

# 3) Build & save KDTree
tree = KDTree(keys)
with open('star_tri_db.pkl', 'wb') as f:
    pickle.dump((keys, vals), f)
print(f"Saved triangle database with {len(keys)} entries.")
