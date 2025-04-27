# ─── constellation_lookup.py ───────────────────────────────────────────
import sys
from skyfield.api import load
from skyfield.data import hipparcos

# we’ll use Astropy’s built‑in constellation boundaries
try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except ImportError:
    print("ERROR: astropy not installed. Run: pip install astropy")
    sys.exit(1)

# 1) Load the full catalog
CATALOG_FILE = 'star_data/hip_main.dat'
with load.open(CATALOG_FILE) as f:
    stars_all = hipparcos.load_dataframe(f)

# 2) Ask for a HIP number (or set it here)
try:
    hip_input = input("Enter a HIP number (e.g. 32349 for Sirius): ").strip()
    hip_id = int(hip_input)
except Exception:
    print("Invalid HIP number.")
    sys.exit(1)

# 3) Lookup RA/Dec for that HIP
if hip_id not in stars_all.index:
    print(f"HIP {hip_id} not found in catalog.")
    sys.exit(1)

ra_deg  = stars_all.at[hip_id, 'ra_degrees']
dec_deg = stars_all.at[hip_id, 'dec_degrees']

# 4) Build a SkyCoord and get the constellation
coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
const = coord.get_constellation()

print(f"HIP {hip_id} ➞ RA {ra_deg:.5f}°, Dec {dec_deg:.5f}°")
print(f"That star lies in the constellation {const}")
