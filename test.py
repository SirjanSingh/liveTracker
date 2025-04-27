import os
import cv2
import math
import numpy as np
import pandas as pd
from math import comb
from itertools import combinations
from scipy.spatial import KDTree
from skyfield.api import load
from skyfield.data import hipparcos

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

def improved_plate_solve(image_path, catalog_file, bright_mag=5.0, blob_params=None, max_triangles=10000):
    """
    Improved plate solver for astronomical images.
    
    Args:
        image_path: Path to the image to analyze
        catalog_file: Path to the Hipparcos catalog file
        bright_mag: Magnitude limit for stars to include in analysis
        blob_params: Dictionary of blob detection parameters (or None for defaults)
        max_triangles: Maximum number of triangles to try
        
    Returns:
        Dictionary containing:
            - transform: Affine transform from image to sky coordinates
            - center_ra_dec: (RA, Dec) of image center
            - detected_stars: List of detected stars with coordinates
            - success: Boolean indicating if plate solving succeeded
    """
    result = {
        'transform': None,
        'center_ra_dec': None,
        'detected_stars': [],
        'success': False
    }
    
    print(f"Starting improved plate solving for {image_path}")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return result
    if not os.path.exists(catalog_file):
        print(f"Error: Catalog file not found at {catalog_file}")
        return result
    
    # Load catalog and filter to bright stars
    try:
        with load.open(catalog_file) as f:
            stars_all = hipparcos.load_dataframe(f)
        stars_all['hip_id'] = stars_all.index
        
        # Filter to bright stars
        stars = stars_all[stars_all['magnitude'] < bright_mag].copy()
        stars.reset_index(drop=True, inplace=True)
        print(f"Using {len(stars)} stars (mag < {bright_mag})")
        
        if len(stars) < 100:
            print("Warning: Very few stars in catalog. Consider increasing bright_mag.")
            if bright_mag < 6.0:
                bright_mag = 6.0
                stars = stars_all[stars_all['magnitude'] < bright_mag].copy()
                stars.reset_index(drop=True, inplace=True)
                print(f"Automatically increased to {len(stars)} stars (mag < {bright_mag})")
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return result
    
    # Process image and detect stars using multiple methods
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"OpenCV couldn't read image at {image_path}")
        
        # Get image dimensions
        height, width = img.shape[:2]
        print(f"Image dimensions: {width}x{height}")
        
        # Convert to grayscale 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try multiple star detection methods and combine results
        all_points = []
        
        # Method 1: Blob detection with different parameters
        blob_detector_params = cv2.SimpleBlobDetector_Params()
        blob_detector_params.minThreshold = 10
        blob_detector_params.maxThreshold = 200
        blob_detector_params.filterByArea = True
        blob_detector_params.minArea = 3
        blob_detector_params.filterByCircularity = True
        blob_detector_params.minCircularity = 0.1
        blob_detector_params.filterByConvexity = True
        blob_detector_params.minConvexity = 0.5
        blob_detector_params.filterByInertia = True
        blob_detector_params.minInertiaRatio = 0.01
        
        detector = cv2.SimpleBlobDetector_create(blob_detector_params)
        keypoints = detector.detect(gray)
        
        print(f"Method 1: Found {len(keypoints)} stars with blob detector")
        for kp in keypoints:
            all_points.append((kp.pt[0], kp.pt[1]))
        
        # Method 2: Adaptive thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple threshold values
        for threshold in [50, 100, 150]:
            _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract centroids from contours
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    all_points.append((cx, cy))
                else:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w*h >= 1:  # Skip very small blobs
                        all_points.append((x + w//2, y + h//2))
            
            print(f"Method 2: Found {len(contours)} additional stars with threshold {threshold}")
        
        # Method 3: Local maxima detection
        kernel_size = 5
        max_filtered = cv2.dilate(blurred, np.ones((kernel_size, kernel_size), np.uint8))
        max_mask = (blurred == max_filtered) & (blurred > np.mean(blurred) + 2*np.std(blurred))
        y_coords, x_coords = np.where(max_mask)
        
        for x, y in zip(x_coords, y_coords):
            all_points.append((x, y))
        
        print(f"Method 3: Found {len(x_coords)} stars with local maxima")
        
        # Consolidate points (merge nearby points)
        if len(all_points) == 0:
            print("Error: No stars detected in image")
            return result
            
        all_points = np.array(all_points)
        
        # Use DBSCAN to cluster nearby points
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=5, min_samples=1).fit(all_points)
        
        # Find cluster centers
        unique_labels = set(clustering.labels_)
        pts = []
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            mask = clustering.labels_ == label
            cluster_points = all_points[mask]
            center = np.mean(cluster_points, axis=0)
            pts.append(center)
        
        pts = np.array(pts)
        print(f"After consolidation: {len(pts)} unique star positions")
        
        if len(pts) < 4:
            print("Error: Not enough star blobs detected after consolidation")
            # Try again with simpler method if the advanced methods failed
            _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    pts.append([cx, cy])
            
            pts = np.array(pts)
            print(f"Emergency detection found {len(pts)} stars")
            
            if len(pts) < 4:
                print("Error: Still not enough stars detected. Try a different image or adjust parameters.")
                return result
    
    except Exception as e:
        print(f"Error during image processing: {e}")
        import traceback
        traceback.print_exc()
        return result
    
    # Precompute 3D vectors for star catalog
    print("Computing 3D vectors for catalog stars...")
    star_vecs = np.stack([
        radec_to_vector(
            math.radians(ra), 
            math.radians(dec)
        )
        for ra, dec in zip(stars['ra_degrees'], stars['dec_degrees'])
    ])
    
    # Build triangle pattern database
    print("Building triangle pattern database...")
    
    # For faster processing, limit the combinations
    max_stars = min(600, len(stars))
    if len(stars) > max_stars:
        print(f"Limiting to {max_stars} stars for faster processing")
        indices = list(range(max_stars))
    else:
        indices = list(range(len(star_vecs)))
    
    # Filter out stars that are too close to each other in the catalog
    min_separation_rad = math.radians(0.1)  # 0.1 degrees minimum separation
    filtered_indices = []
    for i in indices:
        too_close = False
        for j in filtered_indices:
            if angular_dist(star_vecs[i], star_vecs[j]) < min_separation_rad:
                too_close = True
                break
        if not too_close:
            filtered_indices.append(i)
    
    if len(filtered_indices) < len(indices):
        print(f"Filtered catalog to {len(filtered_indices)} stars after removing close pairs")
        indices = filtered_indices
    
    # Build triangles (either all or sample them)
    total_triangles = len(indices) * (len(indices) - 1) * (len(indices) - 2) // 6
    if total_triangles > 1_000_000:
        sampling_factor = int(total_triangles / 1_000_000) + 1
        print(f"Triangle space too large, using 1/{sampling_factor} sampling")
        triangle_indices = []
        for idx, (i, j, k) in enumerate(combinations(indices, 3)):
            if idx % sampling_factor == 0:
                triangle_indices.append((i, j, k))
    else:
        triangle_indices = list(combinations(indices, 3))
    
    print(f"Building {len(triangle_indices)} triangles for pattern database")
    
    # Compute triangle patterns
    db_keys, db_vals = [], []
    for i, j, k in triangle_indices:
        # Calculate side lengths
        dij = angular_dist(star_vecs[i], star_vecs[j])
        djk = angular_dist(star_vecs[j], star_vecs[k])
        dki = angular_dist(star_vecs[k], star_vecs[i])
        ds = np.sort([dij, djk, dki])
        
        # Skip degenerate triangles and very large triangles
        if ds[2] < 1e-6 or ds[2] > math.radians(20):
            continue
            
        # Create normalized triangle pattern
        key = (ds[0]/ds[2], ds[1]/ds[2])
        db_keys.append(key)
        db_vals.append((i, j, k))
    
    # Check if we have enough triangles
    if len(db_keys) < 100:
        print(f"Warning: Only {len(db_keys)} valid triangles found. Results may be unreliable.")
    
    # Build KD-Tree for fast triangle pattern matching
    db_tree = KDTree(db_keys)
    print(f"Triangle database built with {len(db_keys)} entries")
    
    # Start plate solving
    print("Starting plate solving process...")
    transform = None
    best_inliers = 0
    best_match_score = float('inf')
    
    # Limit number of triangles to try
    max_img_triangles = min(max_triangles, len(list(combinations(range(len(pts)), 3))))
    print(f"Will try up to {max_img_triangles} image triangles")
    
    # Filter image points to avoid extremely close ones
    filtered_pts = []
    min_pixel_dist = 5  # Minimum distance between points in pixels
    
    for i, p in enumerate(pts):
        too_close = False
        for fp in filtered_pts:
            if np.linalg.norm(p - fp) < min_pixel_dist:
                too_close = True
                break
        if not too_close:
            filtered_pts.append(p)
    
    if len(filtered_pts) < len(pts):
        print(f"Filtered image points from {len(pts)} to {len(filtered_pts)} after removing close pairs")
        pts = np.array(filtered_pts)
    
    if len(pts) < 4:
        print("Error: Not enough distinct star positions for reliable matching")
        return result
    
    # Try different combinations of 3 points from the image
    progress_step = max(1, max_img_triangles // 10)
    next_progress = progress_step
    
    # Use a minimum triangle size to avoid degenerate configurations
    min_triangle_size = 10  # pixels
    
    for counter, (i, j, k) in enumerate(combinations(range(len(pts)), 3)):
        if counter > max_img_triangles:
            break
            
        # Show progress
        if counter >= next_progress:
            print(f"Progress: {counter}/{max_img_triangles} triangles processed ({counter/max_img_triangles*100:.1f}%)")
            next_progress += progress_step
            
        # Calculate distances between points (in pixels)
        dij = np.linalg.norm(pts[i] - pts[j])
        djk = np.linalg.norm(pts[j] - pts[k])
        dki = np.linalg.norm(pts[k] - pts[i])
        
        # Skip triangles that are too small
        if min(dij, djk, dki) < min_triangle_size:
            continue
        
        # Sort distances for normalization
        ds = np.sort([dij, djk, dki])
        
        # Skip degenerate or nearly degenerate triangles
        if ds[2] < 1e-6 or ds[0]/ds[2] < 0.1:
            continue
            
        # Normalize the triangle to create a pattern
        key = (ds[0]/ds[2], ds[1]/ds[2])
        
        # Query the KD-tree to find similar triangles in our catalog
        # Get more potential matches for better accuracy
        dist, idxs = db_tree.query(key, k=20)
        
        # Try each potential matching triangle from the catalog
        for idx, m in enumerate(np.atleast_1d(idxs)):
            if m >= len(db_vals):
                continue
                
            a, b, c = db_vals[m]
            
            # Set up source (image) and destination (catalog) points
            src = np.float32([pts[i], pts[j], pts[k]])
            dst = np.float32([
                [stars.iloc[a]['ra_degrees'], stars.iloc[a]['dec_degrees']],
                [stars.iloc[b]['ra_degrees'], stars.iloc[b]['dec_degrees']],
                [stars.iloc[c]['ra_degrees'], stars.iloc[c]['dec_degrees']],
            ])
            
            # Compute affine transform from image coordinates to celestial coordinates
            try:
                M, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, 
                                                 ransacReprojThreshold=5.0, 
                                                 maxIters=2000, 
                                                 confidence=0.99)
            except Exception as e:
                print(f"Error in estimateAffine2D: {e}")
                continue
            
            # Check if this is a better match than previous ones
            if M is not None and inliers is not None and inliers.sum() >= 3:
                inlier_count = inliers.sum()
                match_score = dist[idx] / inlier_count  # Lower is better
                
                if inlier_count > best_inliers or (inlier_count == best_inliers and match_score < best_match_score):
                    transform = M
                    best_inliers = inlier_count
                    best_match_score = match_score
                    
                    print(f"Found transform with {inlier_count} inliers, score: {match_score:.6f}")
                    
                    # Verify the transform - map all image points to sky
                    ones = np.ones((len(pts), 1))
                    aug = np.hstack([pts, ones])
                    radec = (M @ aug.T).T
                    
                    # Check if the transform produces valid declination values
                    # Declination must be between -90 and +90 degrees
                    if np.any(radec[:, 1] < -90) or np.any(radec[:, 1] > 90):
                        print("Rejecting transform - produces invalid declination values")
                        continue
                    
                    # If we have a very good match, break early
                    if best_inliers >= 5 and best_match_score < 0.01:
                        break
        
        # Break if we found a very good match
        if best_inliers >= 5 and best_match_score < 0.01:
            break
    
    # Check if we found a valid transformation
    if transform is None:
        print("Plate solve failed: no matching triangle found")
        return result
    
    # Verify and refine the transformation
    print("Verifying and refining the transformation...")
    
    # Map all detected stars to celestial coordinates
    ones = np.ones((len(pts), 1))
    aug = np.hstack([pts, ones])
    radec = (transform @ aug.T).T
    
    # Verify again that all coordinates are valid
    if np.any(radec[:, 1] < -90) or np.any(radec[:, 1] > 90):
        print("Warning: Transform produces some invalid declination values")
        # Attempt to fix by correcting the transform
        for i in range(len(radec)):
            if radec[i, 1] < -90:
                radec[i, 1] = -180 - radec[i, 1]
                radec[i, 0] = (radec[i, 0] + 180) % 360
            elif radec[i, 1] > 90:
                radec[i, 1] = 180 - radec[i, 1]
                radec[i, 0] = (radec[i, 0] + 180) % 360
    
    # Map image center to celestial coordinates
    img_center_x = img.shape[1] / 2
    img_center_y = img.shape[0] / 2
    center_pt = np.array([[img_center_x, img_center_y, 1]])
    ra_dec_center = (transform @ center_pt.T).T[0]
    
    # Normalize RA to [0, 360) range
    ra_dec_center[0] = ra_dec_center[0] % 360
    
    # Verify declination is valid
    if ra_dec_center[1] < -90 or ra_dec_center[1] > 90:
        print(f"Warning: Invalid declination for image center: {ra_dec_center[1]}")
        if ra_dec_center[1] < -90:
            ra_dec_center[1] = -180 - ra_dec_center[1]
            ra_dec_center[0] = (ra_dec_center[0] + 180) % 360
        elif ra_dec_center[1] > 90:
            ra_dec_center[1] = 180 - ra_dec_center[1]
            ra_dec_center[0] = (ra_dec_center[0] + 180) % 360
        print(f"Corrected to: RA={ra_dec_center[0]:.2f}°, Dec={ra_dec_center[1]:.2f}°")
    
    # Create result dictionary
    result = {
        'transform': transform,
        'center_ra_dec': (ra_dec_center[0], ra_dec_center[1]),
        'detected_stars': [(x, y, ra, dec) for (x, y), (ra, dec) in zip(pts, radec)],
        'success': True
    }
    
    # Check for known objects at these coordinates
    pleiades_coords = (56.75, 24.12)
    dist_to_pleiades = math.sqrt((ra_dec_center[0] - pleiades_coords[0])**2 + 
                                 (ra_dec_center[1] - pleiades_coords[1])**2)
    
    if dist_to_pleiades < 15:  # Within 15 degrees
        print(f"This appears to be an image of the Pleiades region!")
    
    print(f"Plate solve successful! Center: RA={ra_dec_center[0]:.2f}°, Dec={ra_dec_center[1]:.2f}°")
    return result



result = improved_plate_solve(
    image_path='synthetic_sky/_DSC3448.jpg',
    catalog_file='star_data/hip_main.dat',
    bright_mag=4.5,
    # blob_threshold=100,
    max_triangles=5000
)

if result['success']:
    print(f"Image center: RA={result['center_ra_dec'][0]:.2f}°, Dec={result['center_ra_dec'][1]:.2f}°")
    
    # Check if this is the Pleiades
    pleiades_coords = (56.75, 24.12)
    center_coords = result['center_ra_dec']
    dist = math.sqrt((center_coords[0] - pleiades_coords[0])**2 + 
                     (center_coords[1] - pleiades_coords[1])**2)
    
    if dist < 15:
        print("This is an image of the Pleiades!")
else:
    print("Plate solving failed")