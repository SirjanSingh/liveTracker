import streamlit as st
import os
import sys
import time
import json
import requests
import tempfile
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import logging
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import threading
import queue

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constellation data
CONSTELLATIONS = {
    'Andromeda': {'abbr': 'And'},
    'Antlia': {'abbr': 'Ant'},
    'Apus': {'abbr': 'Aps'},
    'Aquarius': {'abbr': 'Aqr'},
    'Aquila': {'abbr': 'Aql'},
    'Ara': {'abbr': 'Ara'},
    'Aries': {'abbr': 'Ari'},
    'Auriga': {'abbr': 'Aur'},
    'Boötes': {'abbr': 'Boo'},
    'Caelum': {'abbr': 'Cae'},
    'Camelopardalis': {'abbr': 'Cam'},
    'Cancer': {'abbr': 'Cnc'},
    'Canes Venatici': {'abbr': 'CVn'},
    'Canis Major': {'abbr': 'CMa'},
    'Canis Minor': {'abbr': 'CMi'},
    'Capricornus': {'abbr': 'Cap'},
    'Carina': {'abbr': 'Car'},
    'Cassiopeia': {'abbr': 'Cas'},
    'Centaurus': {'abbr': 'Cen'},
    'Cepheus': {'abbr': 'Cep'},
    'Cetus': {'abbr': 'Cet'},
    'Chamaeleon': {'abbr': 'Cha'},
    'Circinus': {'abbr': 'Cir'},
    'Columba': {'abbr': 'Col'},
    'Coma Berenices': {'abbr': 'Com'},
    'Corona Australis': {'abbr': 'CrA'},
    'Corona Borealis': {'abbr': 'CrB'},
    'Corvus': {'abbr': 'Crv'},
    'Crater': {'abbr': 'Crt'},
    'Crux': {'abbr': 'Cru'},
    'Cygnus': {'abbr': 'Cyg'},
    'Delphinus': {'abbr': 'Del'},
    'Dorado': {'abbr': 'Dor'},
    'Draco': {'abbr': 'Dra'},
    'Equuleus': {'abbr': 'Equ'},
    'Eridanus': {'abbr': 'Eri'},
    'Fornax': {'abbr': 'For'},
    'Gemini': {'abbr': 'Gem'},
    'Grus': {'abbr': 'Gru'},
    'Hercules': {'abbr': 'Her'},
    'Horologium': {'abbr': 'Hor'},
    'Hydra': {'abbr': 'Hya'},
    'Hydrus': {'abbr': 'Hyi'},
    'Indus': {'abbr': 'Ind'},
    'Lacerta': {'abbr': 'Lac'},
    'Leo': {'abbr': 'Leo'},
    'Leo Minor': {'abbr': 'LMi'},
    'Lepus': {'abbr': 'Lep'},
    'Libra': {'abbr': 'Lib'},
    'Lupus': {'abbr': 'Lup'},
    'Lynx': {'abbr': 'Lyn'},
    'Lyra': {'abbr': 'Lyr'},
    'Mensa': {'abbr': 'Men'},
    'Microscopium': {'abbr': 'Mic'},
    'Monoceros': {'abbr': 'Mon'},
    'Musca': {'abbr': 'Mus'},
    'Norma': {'abbr': 'Nor'},
    'Octans': {'abbr': 'Oct'},
    'Ophiuchus': {'abbr': 'Oph'},
    'Orion': {'abbr': 'Ori'},
    'Pavo': {'abbr': 'Pav'},
    'Pegasus': {'abbr': 'Peg'},
    'Perseus': {'abbr': 'Per'},
    'Phoenix': {'abbr': 'Phe'},
    'Pictor': {'abbr': 'Pic'},
    'Pisces': {'abbr': 'Psc'},
    'Piscis Austrinus': {'abbr': 'PsA'},
    'Puppis': {'abbr': 'Pup'},
    'Pyxis': {'abbr': 'Pyx'},
    'Reticulum': {'abbr': 'Ret'},
    'Sagitta': {'abbr': 'Sge'},
    'Sagittarius': {'abbr': 'Sgr'},
    'Scorpius': {'abbr': 'Sco'},
    'Sculptor': {'abbr': 'Scl'},
    'Scutum': {'abbr': 'Sct'},
    'Serpens': {'abbr': 'Ser'},
    'Sextans': {'abbr': 'Sex'},
    'Taurus': {'abbr': 'Tau'},
    'Telescopium': {'abbr': 'Tel'},
    'Triangulum': {'abbr': 'Tri'},
    'Triangulum Australe': {'abbr': 'TrA'},
    'Tucana': {'abbr': 'Tuc'},
    'Ursa Major': {'abbr': 'UMa'},
    'Ursa Minor': {'abbr': 'UMi'},
    'Vela': {'abbr': 'Vel'},
    'Virgo': {'abbr': 'Vir'},
    'Volans': {'abbr': 'Vol'},
    'Vulpecula': {'abbr': 'Vul'}
}

# Helper function for constellation boundaries (simplified for this demo)
# In a real app, you'd use actual constellation boundary data
def get_constellation_for_coords(ra, dec):
    # This is a simplified placeholder - in reality, you would use actual constellation boundary data
    # For demo purposes, we'll return a constellation based on rough RA/Dec regions
    
    # Normalize RA to 0-24 hours
    ra_hours = ra / 15.0  # Convert degrees to hours
    
    # Very simplified mapping based on rough RA regions
    if 0 <= ra_hours < 2:
        return "Pisces"
    elif 2 <= ra_hours < 4:
        return "Aries"
    elif 4 <= ra_hours < 6:
        return "Taurus"
    elif 6 <= ra_hours < 8:
        return "Gemini"
    elif 8 <= ra_hours < 10:
        return "Cancer" if dec > 0 else "Hydra"
    elif 10 <= ra_hours < 12:
        return "Leo" if dec > 0 else "Hydra"
    elif 12 <= ra_hours < 14:
        return "Virgo" if dec > 0 else "Centaurus"
    elif 14 <= ra_hours < 16:
        return "Libra" if dec > 0 else "Lupus"
    elif 16 <= ra_hours < 18:
        return "Scorpius" if dec > -30 else "Sagittarius"
    elif 18 <= ra_hours < 20:
        return "Sagittarius"
    elif 20 <= ra_hours < 22:
        return "Capricornus" if dec > -30 else "Piscis Austrinus"
    else:  # 22-24
        return "Aquarius" if dec > -30 else "Cetus"

class AstrometryNetClient:
    """Client for interacting with the Astrometry.net API"""
    
    def __init__(self, api_key=None):
        """Initialize the client with API key"""
        self.api_key = api_key
        self.session = requests.session()
        self.base_url = "http://nova.astrometry.net/api/"
        # Set longer timeouts to avoid connection issues
        self.timeout = 100
    
    def login(self):
        """Log in to the Astrometry.net API and get a session key"""
        data = {
            'request-json': json.dumps({"apikey": self.api_key})
        }
        try:
            response = self.session.post(self.base_url + 'login', data=data, timeout=self.timeout)
            result = response.json()
            
            if result.get('status') != 'success':
                logger.error(f"Login failed: {result}")
                return None
            
            logger.info("Successfully logged in to Astrometry.net")
            self.session_key = result.get('session')
            return self.session_key
        except requests.exceptions.RequestException as e:
            logger.error(f"Login request failed: {e}")
            return None
    
    def upload_image(self, img_data, **kwargs):
        """Upload an image to be solved"""
        try:
            # Get image dimensions from the data
            img = Image.open(BytesIO(img_data))
            width, height = img.size
            
            # Set up submission parameters
            params = {
                'session': self.session_key,
                'allow_commercial_use': 'd', # default
                'allow_modifications': 'd', # default
                'publicly_visible': 'n', # private
                'scale_units': 'degwidth', # use degrees for scale
                'scale_type': 'ul', # bounds for scale
                'scale_lower': 0.1, # lower bound of scale in degrees
                'scale_upper': 180, # upper bound of scale in degrees
                'center_ra': kwargs.get('center_ra', None), # approximate RA in degrees
                'center_dec': kwargs.get('center_dec', None), # approximate Dec in degrees
                'radius': kwargs.get('radius', None), # search radius in degrees
                'downsample_factor': kwargs.get('downsample_factor', 2), # downsample for speed
                'tweak_order': kwargs.get('tweak_order', 2), # polynomial order for distortion
                'image_width': width,
                'image_height': height,
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Create JSON request
            json_data = json.dumps(params)
            
            logger.info("Uploading image to Astrometry.net...")
            response = self.session.post(
                self.base_url + 'upload',
                files={
                    'request-json': (None, json_data),
                    'file': ('image.fits', img_data, 'application/octet-stream')
                },
                timeout=self.timeout
            )
            
            result = response.json()
            
            if result.get('status') != 'success':
                logger.error(f"Upload failed: {result}")
                return None
                
            submission_id = result.get('subid')
            logger.info(f"Image uploaded successfully. Submission ID: {submission_id}")
            return submission_id
        except requests.exceptions.RequestException as e:
            logger.error(f"Upload request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            return None
    
    def check_submission_status(self, submission_id):
        """Check the status of a submission with retry logic"""
        max_retries = 5
        retry_delay = 3
        
        data = {
            'request-json': json.dumps({
                'apikey': self.api_key,
                'job_id': submission_id,
            })
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self.base_url + 'submissions/' + str(submission_id), 
                    data=data,
                    timeout=self.timeout
                )
                result = response.json()
                return result
            except (requests.exceptions.ChunkedEncodingError, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout) as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Create a new session to avoid persistent connection issues
                    self.session = requests.session()
                else:
                    logger.error("Maximum retry attempts reached.")
                    # Instead of failing, return a status that indicates we should check
                    # the job status by another means
                    return {"status": "retry_needed", "message": str(e)}
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Maximum retry attempts reached.")
                    return {"status": "retry_needed", "message": "Invalid JSON response"}
                    
    def check_job_directly(self, submission_id):
        """Alternative way to check job status when the submissions endpoint fails"""
        try:
            # Try to get jobs for this submission directly from the website
            response = self.session.get(
                f"http://nova.astrometry.net/status/{submission_id}",
                timeout=self.timeout
            )
            
            # Parse the HTML response to find job IDs
            # This is a simple approach and might break if the website changes
            if "This submission has been processed" in response.text:
                # Look for job ID in the HTML
                import re
                job_ids = re.findall(r'Jobs: <a href="/status/(\d+)">', response.text)
                
                if job_ids:
                    logger.info(f"Found job ID: {job_ids[0]} for submission {submission_id}")
                    return {
                        "processing_finished": True,
                        "jobs": [int(job_ids[0])]
                    }
            
            # Check if it's still processing
            if "We're still working on your submission" in response.text:
                return {
                    "processing_finished": False,
                    "processing_status": "Still processing"
                }
                
            return {
                "processing_finished": False,
                "processing_status": "Status unclear, please check manually at nova.astrometry.net"
            }
            
        except Exception as e:
            logger.error(f"Error checking job directly: {e}")
            return {
                "processing_finished": False,
                "processing_status": f"Error: {str(e)}"
            }
    
    def wait_for_results(self, submission_id, status_queue, timeout=600, interval=5):
        """Wait for submission to complete and return the job ID"""
        logger.info("Waiting for plate solving to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                status = self.check_submission_status(submission_id)
                
                # Handle retry needed status
                if status.get("status") == "retry_needed":
                    logger.info("Trying alternative method to check job status...")
                    status = self.check_job_directly(submission_id)
                
                # Update the queue with status
                status_queue.put({
                    "status": status.get('processing_status', 'Processing...'),
                    "finished": status.get('processing_finished', False),
                    "progress": min(90, int((time.time() - start_time) / timeout * 100))
                })
                
                if status.get('processing_finished'):
                    jobs = status.get('jobs', [])
                    if jobs:
                        logger.info(f"Plate solving completed. Job ID: {jobs[0]}")
                        status_queue.put({
                            "status": "Completed!",
                            "finished": True,
                            "progress": 100,
                            "job_id": jobs[0]
                        })
                        return jobs[0]
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error checking status: {e}")
                status_queue.put({
                    "status": f"Error: {str(e)}",
                    "finished": False,
                    "progress": min(90, int((time.time() - start_time) / timeout * 100))
                })
                time.sleep(interval)
                
        logger.error(f"Timeout after {timeout} seconds")
        status_queue.put({
            "status": "Timeout! Check manually.",
            "finished": True,
            "progress": 100,
            "timeout": True
        })
        return None
    
    def get_results(self, job_id):
        """Get the results of a completed job"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                data = {
                    'request-json': json.dumps({
                        'apikey': self.api_key,
                    })
                }
                
                response = self.session.post(
                    self.base_url + 'jobs/' + str(job_id) + '/info', 
                    data=data,
                    timeout=10
                )
                result = response.json()
                return result
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                logger.warning(f"Error getting results (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    logger.error("Failed to get results after retries.")
                    return {"status": "error", "message": str(e)}
    
    def download_wcs_file(self, job_id):
        """Download the WCS file for a solved image"""
        try:
            url = f"http://nova.astrometry.net/wcs_file/{job_id}"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Failed to download WCS file: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading WCS file: {e}")
            return None
    
    def download_annotated_image(self, job_id):
        """Download the annotated image"""
        try:
            url = f"http://nova.astrometry.net/annotated_display/{job_id}"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Failed to download annotated image: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading annotated image: {e}")
            return None

def extract_constellation_info(wcs_data):
    """Extract constellation information from WCS data"""
    return 
    try:
        # Create a temporary file to save the WCS data
        with tempfile.NamedTemporaryFile(suffix='.wcs', delete=False) as tmp:
            tmp.write(wcs_data)
            tmp_path = tmp.name
        
        # Load the WCS data
        w = WCS(tmp_path)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        # Get the center of the image
        height, width = w.array_shape
        center_x, center_y = width // 2, height // 2
        
        # Convert pixel coordinates to world coordinates
        ra, dec = w.all_pix2world(center_x, center_y, 0)
        
        # Get constellation for these coordinates
        constellation = get_constellation_for_coords(ra, dec)
        
        # Format coordinates
        coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        ra_formatted = coords.ra.to_string(unit=u.hourangle, sep=':', precision=1)
        dec_formatted = coords.dec.to_string(sep=':', precision=1)
        
        return {
            'ra': ra,
            'dec': dec,
            'ra_formatted': ra_formatted,
            'dec_formatted': dec_formatted,
            'constellation': constellation,
            'constellation_abbr': CONSTELLATIONS.get(constellation, {}).get('abbr', '')
        }
    except Exception as e:
        logger.error(f"Error extracting constellation info: {e}")
        return {
            'ra': None,
            'dec': None,
            'ra_formatted': 'Unknown',
            'dec_formatted': 'Unknown',
            'constellation': 'Unknown',
            'constellation_abbr': ''
        }

def format_results_for_display(results):
    """Format the plate solving results for display"""
    if not results or results.get('status') != 'success':
        return "No results available or processing failed."
    
    # Extract calibration data
    cal = results.get('calibration', {})
    ra = cal.get('ra')
    dec = cal.get('dec')
    orientation = cal.get('orientation')
    pixscale = cal.get('pixscale')
    radius = cal.get('radius')
    
    # Format coordinates
    if ra is not None and dec is not None:
        coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        ra_formatted = coords.ra.to_string(unit=u.hourangle, sep=':', precision=1)
        dec_formatted = coords.dec.to_string(sep=':', precision=1)
    else:
        ra_formatted = "Unknown"
        dec_formatted = "Unknown"
    
    # Format objects in field
    objects = results.get('objects_in_field', [])
    objects_text = ", ".join(objects[:5])
    if len(objects) > 5:
        objects_text += f" and {len(objects) - 5} more"
    
    # Constellation (would come from the WCS data in a real implementation)
    constellation = get_constellation_for_coords(ra or 0, dec or 0)
    
    result_text = f"""
    ### Plate Solving Results
    
    **Center coordinates:** {ra_formatted}, {dec_formatted}
    
    **Field of view:** {radius:.2f}° radius
    
    **Image scale:** {pixscale:.2f} arcsec/pixel
    
    **Orientation:** {orientation:.1f}°
    
    **Constellation:** {constellation}
    
    **Notable objects:** {objects_text if objects else "None identified"}
    """
    
    return result_text

def start_solver(api_key, image_bytes, status_queue):
    try:
        # Initialize client
        client = AstrometryNetClient(api_key)
        if not client.login():
            status_queue.put({"status": "Login failed", "finished": True, "progress": 100, "error": True})
            return
        
        # Upload image
        status_queue.put({"status": "Uploading image...", "progress": 10})
        submission_id = client.upload_image(image_bytes)
        if not submission_id:
            status_queue.put({"status": "Upload failed", "finished": True, "progress": 100, "error": True})
            return
        
        # Wait for results
        job_id = client.wait_for_results(submission_id, status_queue)
        if not job_id:
            return
        
        # Get results
        status_queue.put({"status": "Downloading results...", "progress": 95})
        results = client.get_results(job_id)
        
        # Download WCS file and annotated image
        wcs_data = client.download_wcs_file(job_id)
        annotated_image = client.download_annotated_image(job_id)
        # Save offline
        if wcs_data:
            with open(f'wcs_solution_{job_id}.wcs', 'wb') as f:
                f.write(wcs_data)
            logger.info(f"WCS file saved as wcs_solution_{job_id}.wcs")

        if annotated_image:
            with open(f'annotated_image_{job_id}.jpg', 'wb') as f:
                f.write(annotated_image)
            logger.info(f"Annotated image saved as annotated_image_{job_id}.jpg")

        # Extract constellation information
        constellation_info = extract_constellation_info(wcs_data) if wcs_data else None
        
        # Send back complete results
        status_queue.put({
            "status": "Complete!",
            "finished": True,
            "progress": 100,
            "results": results,
            "wcs_data": base64.b64encode(wcs_data).decode('utf-8') if wcs_data else None,
            "annotated_image": base64.b64encode(annotated_image).decode('utf-8') if annotated_image else None,
            "constellation_info": constellation_info,
            "job_id": job_id
        })
        
    except Exception as e:
        logger.error(f"Error in solver thread: {e}")
        status_queue.put({"status": f"Error: {str(e)}", "finished": True, "progress": 100, "error": True})

def get_example_images():
    """Return a list of example astronomical images for testing"""
    return [
        "https://apod.nasa.gov/apod/image/2304/M31_HubbleSpitzerSchmidt_1019.jpg",
        "https://apod.nasa.gov/apod/image/2304/ngc253_chakrabarti_5939.jpg",
        "https://apod.nasa.gov/apod/image/2304/OrionDeep_Ropert_3646.jpg"
    ]

def load_lottie_animation(url):
    """Load a Lottie animation from URL"""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    st.set_page_config(
        page_title="Astrometry Plate Solver", 
        page_icon="🔭", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for animations and styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4f8bf9;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .upload-container {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .results-container {
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #f9f9f9;
    }
    .st-emotion-cache-1kyxreq {
        display: flex;
        justify-content: center;
    }
    .uploaded-img {
        max-height: 400px;
        margin: 0 auto;
        display: block;
    }
    </style>
    <div class="main-header">🔭 AstroPlate Solver</div>
    <div class="subheader">Upload an astronomical image to identify its sky position and contents</div>
    """, unsafe_allow_html=True)
    
    # Sidebar with API key input
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Astrometry.net API Key", type="password")
        st.markdown("Don't have an API key? [Get one here](http://nova.astrometry.net/api_key)")
        
        st.subheader("Advanced Options")
        timeout = st.slider("Solving timeout (seconds)", 60, 600, 300)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses the Astrometry.net service to solve astronomical images.
        
        Plate solving identifies the exact position of your image in the night sky, 
        including coordinates, field of view, and objects captured.
        
        Created with ❤️ by Sirjan Singh
        """)
    
    # Initialize session state
    if 'job_running' not in st.session_state:
        st.session_state.job_running = False
    if 'status_queue' not in st.session_state:
        st.session_state.status_queue = queue.Queue()
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None
    if 'wcs_data' not in st.session_state:
        st.session_state.wcs_data = None
    if 'constellation_info' not in st.session_state:
        st.session_state.constellation_info = None
    
    # File uploader
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an astronomical image...", 
                                         type=["jpg", "jpeg", "png", "fits", "fit"],
                                         help="Upload your astronomical image to be plate solved")
        
        # Add example image selector
        st.markdown("Or try one of our example images:")
        example_images = get_example_images()
        example_selection = st.selectbox("Select example", 
                                         [""] + [f"Example {i+1}" for i in range(len(example_images))],
                                         format_func=lambda x: "Select an example" if x == "" else x)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Solve button and progress
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        solve_button = st.button("🚀 Solve Plate", 
                                 type="primary", 
                                 disabled=st.session_state.job_running or (not uploaded_file and example_selection == "") or not api_key,
                                 help="Start the plate solving process")
        
        if st.session_state.job_running:
            # Show progress
            status_container = st.empty()
            progress_bar = st.progress(0)
            
           # Check for updates from the status queue
            try:
                latest_status = st.session_state.status_queue.get_nowait()
                    
                    # Update progress and status
                status_container.markdown(f"**Status:** {latest_status.get('status', 'Processing...')}")
                progress_bar.progress(latest_status.get('progress', 0))
                   
                    # Check if the job is finished
                if latest_status.get('finished', False):
                    st.session_state.job_running = False
                        
                        # Save results if available
                    if 'results' in latest_status:
                        st.session_state.results = latest_status['results']
                    if 'annotated_image' in latest_status:
                        st.session_state.annotated_image = latest_status['annotated_image']
                    if 'wcs_data' in latest_status:
                        st.session_state.wcs_data = latest_status['wcs_data']
                    if 'constellation_info' in latest_status:
                        st.session_state.constellation_info = latest_status['constellation_info']
                        
                        # Show success or error message
                    if latest_status.get('error', False):
                        st.error("An error occurred during plate solving. Please try again.")
                    elif latest_status.get('timeout', False):
                        st.warning("The operation timed out. Try again or check manually.")
                    else:
                        st.success("Plate solving completed successfully!")
                            
                        # Force a rerun to show results
                    st.rerun()
            except queue.Empty:
                    # No update yet, continue
                pass
    
    # Load and display example image if selected
    if example_selection != "" and not uploaded_file:
        example_idx = int(example_selection.split(" ")[1]) - 1
        if 0 <= example_idx < len(example_images):
            try:
                response = requests.get(example_images[example_idx])
                img_bytes = response.content
                st.session_state.uploaded_image = img_bytes
                uploaded_file = BytesIO(img_bytes)  # Create a file-like object
                uploaded_file.name = f"example_{example_idx+1}.jpg"
                st.success(f"Loaded example image {example_idx+1}")
            except Exception as e:
                st.error(f"Error loading example image: {str(e)}")
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            # Read the image file
            img_bytes = uploaded_file.getvalue()
            st.session_state.uploaded_image = img_bytes
            
            # Display the uploaded image
            if uploaded_file.name.lower().endswith(('.fits', '.fit')):
                # Handle FITS files
                with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                
                # Load FITS data
                with fits.open(tmp_path) as hdul:
                    image_data = hdul[0].data
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
                # Create a matplotlib figure
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(image_data, cmap='gray')
                ax.set_title("Uploaded FITS Image")
                ax.axis('off')
                
                # Display the plot
                st.pyplot(fig)
            else:
                # Handle JPG, PNG, etc.
                img = Image.open(BytesIO(img_bytes))
                st.image(img, caption="Uploaded Image", use_column_width=True, clamp=True)
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
    
    # Start solving process if button is clicked
    if solve_button and not st.session_state.job_running:
        if not api_key:
            st.error("Please enter your Astrometry.net API key first!")
        elif uploaded_file is None and example_selection == "":
            st.error("Please upload an image or select an example first!")
        else:
            # Start the solving thread
            st.session_state.job_running = True
            st.session_state.status_queue = queue.Queue()
            
            # Get the image bytes
            img_bytes = st.session_state.uploaded_image
            
            # Create and start the solver thread
            solver_thread = threading.Thread(
                target=start_solver,
                args=(api_key, img_bytes, st.session_state.status_queue)
            )
            solver_thread.daemon = True
            solver_thread.start()
            
            # Force a rerun to show progress
            st.rerun()
    
    # Display results if available
    if st.session_state.results is not None:
        st.markdown("---")
        st.markdown("## Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            if st.session_state.uploaded_image:
                # Display the original image
                try:
                    if uploaded_file and uploaded_file.name.lower().endswith(('.fits', '.fit')):
                        # Create a matplotlib figure for FITS
                        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
                            tmp.write(st.session_state.uploaded_image)
                            tmp_path = tmp.name
                        
                        with fits.open(tmp_path) as hdul:
                            image_data = hdul[0].data
                        
                        os.unlink(tmp_path)
                        
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(image_data, cmap='gray')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        # Display regular image
                        img = Image.open(BytesIO(st.session_state.uploaded_image))
                        st.image(img, use_column_width=True, clamp=True)
                except Exception as e:
                    st.error(f"Error displaying original image: {str(e)}")
        
        with col2:
            st.markdown("### Solved Image with Annotations")
            if st.session_state.annotated_image:
                try:
                    # Display the annotated image
                    annotated_bytes = base64.b64decode(st.session_state.annotated_image)
                    annotated_img = Image.open(BytesIO(annotated_bytes))
                    st.image(annotated_img, use_column_width=True, clamp=True)
                except Exception as e:
                    st.error(f"Error displaying annotated image: {str(e)}")
                    
        # Display constellation information
        if st.session_state.constellation_info:
            st.markdown("---")
            st.markdown("## Astrometric Data")
            
            info = st.session_state.constellation_info
            
            # Create a nice info box
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.markdown("### Sky Position")
                
                # Create a nice table for coordinates
                coord_data = {
                    "Property": ["Right Ascension", "Declination"],
                    "Value": [info.get('ra_formatted', 'Unknown'), info.get('dec_formatted', 'Unknown')]
                }
                
                st.table(pd.DataFrame(coord_data))
                
                # Display additional data from the results
                if st.session_state.results and st.session_state.results.get('status') == 'success':
                    cal = st.session_state.results.get('calibration', {})
                    
                    # Field of view and pixel scale
                    fov_data = {
                        "Property": ["Field of View", "Pixel Scale", "Orientation"],
                        "Value": [
                            f"{cal.get('radius', 'Unknown')}°",
                            f"{cal.get('pixscale', 'Unknown')} arcsec/pixel",
                            f"{cal.get('orientation', 'Unknown')}°"
                        ]
                    }
                    
                    st.markdown("### Field of View Data")
                    st.table(pd.DataFrame(fov_data))
            
            with info_col2:
                st.markdown("### Constellation")
                
                # Create a box with constellation info
                constellation = info.get('constellation', 'Unknown')
                abbr = info.get('constellation_abbr', '')
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #f5f5f5; text-align: center;">
                    <h1 style="margin: 0; color: #4f8bf9;">{constellation}</h1>
                    <p style="color: #6c757d;">{abbr}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display objects in field if available
                if st.session_state.results:
                    objects = st.session_state.results.get('objects_in_field', [])
                    if objects:
                        st.markdown("### Notable Objects in Field")
                        for obj in objects[:10]:  # Limit to 10 objects
                            st.markdown(f"- {obj}")
                        
                        if len(objects) > 10:
                            st.markdown(f"*... and {len(objects) - 10} more*")
        
        # Show link to Astrometry.net
        if 'job_id' in st.session_state.results:
            job_id = st.session_state.results['job_id']
            st.markdown("---")
            st.markdown(f"""
            ### View on Astrometry.net
            
            [Click here to view full details on Astrometry.net](http://nova.astrometry.net/status/{job_id})
            """)
        
        # Option to download WCS file
        if st.session_state.wcs_data:
            st.download_button(
                label="Download WCS File",
                data=base64.b64decode(st.session_state.wcs_data),
                file_name="astrometry_solution.wcs",
                mime="application/octet-stream"
            )
        
        # Button to clear results and start over
        if st.button("Start Over"):
            # Clear all results
            st.session_state.results = None
            st.session_state.uploaded_image = None
            st.session_state.annotated_image = None
            st.session_state.wcs_data = None
            st.session_state.constellation_info = None
            st.rerun()

# Add a function to display animations during processing
def display_loading_animation():
    """Display a loading animation while processing"""
    # Use a Lottie animation or simple CSS animation
    animation_html = """
    <div class="loading-animation">
        <style>
        .loading-animation {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
        }
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 5px solid rgba(0, 0, 0, 0.1);
            border-radius: 50%;
            border-top-color: #4F8BF9;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        </style>
        <div class="loading-spinner"></div>
    </div>
    """
    st.markdown(animation_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()