import os
import cv2
import math
import time
import numpy as np
from itertools import combinations
from math import comb
from skyfield.api import load
from skyfield.data import hipparcos
from scipy.spatial import KDTree

# ─── CONFIG ─────────────────────────────────────────────────────────
IMAGE_PATH    = 'synthetic_sky/_DSC3448.jpg'
CATALOG_FILE  = 'star_data/hip_main.dat'
OUTPUT_IMAGE  = 'results/annotated_offline.jpg'
BRIGHT_MAG    = 3.0      # filter Hipparcos to mag < this
SAMPLE_TRIS   = 50000    # how many triangles to time for estimate

# ─── UTILS ──────────────────────────────────────────────────────────
def angular_dist(p, q):
    """Angular distance in radians between two unit vectors."""
    return np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))

def radec_to_vector(ra_rad, dec_rad):
    """Convert RA/Dec (radians) to 3D unit vector."""
    return np.array([
        math.cos(dec_rad)*math.cos(ra_rad),
        math.cos(dec_rad)*math.sin(ra_rad),
        math.sin(dec_rad),
    ])

def ask_run_time_estimate(stars_df, sample_tris=SAMPLE_TRIS):
    N = len(stars_df)
    total = comb(N, 3)
    print(f"\nMag < {BRIGHT_MAG} ⇒ {N} stars ⇒ {total:,} triangles total.")
    if total < sample_tris:
        sample = total
    else:
        sample = sample_tris
    print(f"Sampling {sample} triangles to estimate build time...", end='', flush=True)
    # precompute vectors
    vecs = [radec_to_vector(math.radians(ra), math.radians(dec))
            for ra, dec in zip(stars_df['ra_degrees'], stars_df['dec_degrees'])]
    start = time.time()
    for _, (i, j, k) in zip(range(int(sample)), combinations(range(N), 3)):
        dij = angular_dist(vecs[i], vecs[j])
        djk = angular_dist(vecs[j], vecs[k])
        dki = angular_dist(vecs[k], vecs[i])
        ds = sorted([dij, djk, dki])
        _ = (ds[0]/ds[2], ds[1]/ds[2])
    elapsed = time.time() - start
    t_per = elapsed / sample
    est = t_per * total
    print(f" done in {elapsed:.2f}s ({t_per:.3e}s/triangle).")
    print(f"→ Estimated full build: {est/60:.1f} minutes.\n")
    ans = input("Proceed with full triangle DB build? (y/N): ").strip().lower()
    return ans == 'y'

# ─── 1) LOAD & FILTER CATALOG ─────────────────────────────────────────
with load.open(CATALOG_FILE) as f:
    stars = hipparcos.load_dataframe(f)
stars = stars[stars['magnitude'] < BRIGHT_MAG].reset_index(drop=True)
print(f"[+] Catalog filtered to {len(stars)} bright stars (mag < {BRIGHT_MAG})")

# ─── 2) ASK BEFORE BUILDING TRIANGLE DB ───────────────────────────────
if not ask_run_time_estimate(stars):
    print("Build aborted by user.")
    exit()

# ─── 3) PRECOMPUTE 3D VECTORS FOR TRIANGLE DB ─────────────────────────
star_vecs = np.stack([
    radec_to_vector(math.radians(ra), math.radians(dec))
    for ra, dec in zip(stars['ra_degrees'], stars['dec_degrees'])
])

# ─── 4) BUILD TRIANGLE DATABASE ────────────────────────────────────────
db_keys, db_vals = [], []
for i, j, k in combinations(range(len(star_vecs)), 3):
    dij = angular_dist(star_vecs[i], star_vecs[j])
    djk = angular_dist(star_vecs[j], star_vecs[k])
    dki = angular_dist(star_vecs[k], star_vecs[i])
    ds = np.sort([dij, djk, dki])
    if ds[2] == 0:
        continue
    db_keys.append((ds[0]/ds[2], ds[1]/ds[2]))
    db_vals.append((i, j, k))
db_tree = KDTree(db_keys)
print(f"[+] Built triangle‐pattern DB with {len(db_keys)} entries")

# ─── 5) LOAD IMAGE & DETECT BLOBS ──────────────────────────────────────
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

pts = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w*h < 5:
        continue
    pts.append([x + w//2, y + h//2])
pts = np.array(pts, dtype=float)
print(f"[+] Detected {len(pts)} star‐blobs in the image")

# ─── 6) PLATE‐SOLVE via TRIANGLE MATCHING ───────────────────────────────
transform = None
for (i, j, k) in combinations(range(len(pts)), 3):
    dij = np.linalg.norm(pts[i]-pts[j])
    djk = np.linalg.norm(pts[j]-pts[k])
    dki = np.linalg.norm(pts[k]-pts[i])
    ds = np.sort([dij, djk, dki])
    if ds[2] == 0:
        continue
    key = (ds[0]/ds[2], ds[1]/ds[2])
    dist, idxs = db_tree.query(key, k=3)
    for m in np.atleast_1d(idxs):
        a, b, c = db_vals[m]
        src = np.float32([pts[i], pts[j], pts[k]])
        dst = np.float32([
            [stars.at[a, 'ra_degrees'], stars.at[a, 'dec_degrees']],
            [stars.at[b, 'ra_degrees'], stars.at[b, 'dec_degrees']],
            [stars.at[c, 'ra_degrees'], stars.at[c, 'dec_degrees']],
        ])
        M, inliers = cv2.estimateAffine2D(src, dst)
        if M is not None and inliers is not None and inliers.sum() >= 2:
            transform = M
            print(f"[+] Found transform with triangle ({i},{j},{k}) -> ({a},{b},{c})")
            break
    if transform is not None:
        break
if transform is None:
    raise RuntimeError("Plate solve failed: no triangle match found")

# ─── 7) ANNOTATE ALL BLOBS ───────────────────────────────────────────────
ones = np.ones((len(pts),1))
aug  = np.hstack([pts, ones])
radec = (transform @ aug.T).T

full_coords = np.column_stack((stars['ra_degrees'], stars['dec_degrees']))
full_tree   = KDTree(full_coords)

for (x, y), (ra, dec) in zip(pts, radec):
    _, idx = full_tree.query([ra, dec])
    star    = stars.iloc[idx]
    name    = star.get('proper') or f"HIP {star.name}"
    cv2.circle(img, (int(x), int(y)), 5, (0,255,0), 1)
    cv2.putText(img, name, (int(x)+5, int(y)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

# ─── 8) SAVE & DISPLAY ───────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
cv2.imwrite(OUTPUT_IMAGE, img)
print(f"[+] Annotated image saved to {OUTPUT_IMAGE}")

cv2.namedWindow("Solved & Annotated", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Solved & Annotated", 800, 600)
cv2.imshow("Solved & Annotated", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
