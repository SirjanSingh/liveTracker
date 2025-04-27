#!/usr/bin/env python3
"""
Astrometry.net API Plate Solver

This script allows you to upload an astronomical image to the Astrometry.net service
and retrieve the plate solution containing celestial coordinates and other data.

Requirements:
- requests
- pillow (PIL)

To install dependencies:
pip install requests pillow
"""

import os
import sys
import time
import json
import requests
from io import BytesIO
from PIL import Image
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AstrometryNetClient:
    """Client for interacting with the Astrometry.net API"""
    
    def __init__(self, api_key=None):
        """Initialize the client with API key"""
        self.api_key = api_key
        self.session = requests.session()
        self.base_url = "http://nova.astrometry.net/api/"
        # Set longer timeouts to avoid connection issues
        self.timeout = 60
    
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
                sys.exit(1)
            
            logger.info("Successfully logged in to Astrometry.net")
            self.session_key = result.get('session')
            return self.session_key
        except requests.exceptions.RequestException as e:
            logger.error(f"Login request failed: {e}")
            sys.exit(1)
    
    def upload_image(self, image_path, **kwargs):
        """Upload an image to be solved"""
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
            
            # Get image dimensions
            with Image.open(image_path) as img:
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
                sys.exit(1)
                
            submission_id = result.get('subid')
            logger.info(f"Image uploaded successfully. Submission ID: {submission_id}")
            return submission_id
        except requests.exceptions.RequestException as e:
            logger.error(f"Upload request failed: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            sys.exit(1)
    
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
    
    def wait_for_results(self, submission_id, timeout=600, interval=10):
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
                
                if status.get('processing_finished'):
                    jobs = status.get('jobs', [])
                    if jobs:
                        logger.info(f"Plate solving completed. Job ID: {jobs[0]}")
                        return jobs[0]
                
                status_msg = status.get('processing_status', 'unknown')
                logger.info(f"Status: {status_msg}")
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error checking status: {e}")
                time.sleep(interval)
                
        logger.error(f"Timeout after {timeout} seconds")
        logger.info("You can check your submission status manually at:")
        logger.info(f"http://nova.astrometry.net/status/{submission_id}")
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
                    timeout=self.timeout
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
    
    def download_wcs_file(self, job_id, output_dir='.'):
        """Download the WCS file for a solved image"""
        try:
            url = f"http://nova.astrometry.net/wcs_file/{job_id}"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                output_path = os.path.join(output_dir, f"{job_id}.wcs")
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"WCS file saved to {output_path}")
                return output_path
            else:
                logger.error(f"Failed to download WCS file: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading WCS file: {e}")
            return None
    
    def download_annotated_image(self, job_id, output_dir='.'):
        """Download the annotated image"""
        try:
            url = f"http://nova.astrometry.net/annotated_display/{job_id}"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                output_path = os.path.join(output_dir, f"{job_id}_annotated.jpg")
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Annotated image saved to {output_path}")
                return output_path
            else:
                logger.error(f"Failed to download annotated image: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading annotated image: {e}")
            return None

def format_results(results):
    """Format the plate solving results in a readable way"""
    if not results:
        return "No results available"
    
    output = []
    output.append("=== PLATE SOLVING RESULTS ===")
    
    # Basic information
    output.append(f"Status: {results.get('status', 'unknown')}")
    
    if results.get('status') != 'success':
        return "\n".join(output)
    
    # Calibration information
    calibration = results.get('calibration', {})
    if calibration:
        output.append("\n=== CALIBRATION ===")
        output.append(f"RA (center): {calibration.get('ra', 'N/A'):.6f} degrees")
        output.append(f"Dec (center): {calibration.get('dec', 'N/A'):.6f} degrees")
        output.append(f"Orientation: {calibration.get('orientation', 'N/A'):.2f} degrees")
        output.append(f"Pixel scale: {calibration.get('pixscale', 'N/A'):.4f} arcsec/pixel")
        output.append(f"Radius: {calibration.get('radius', 'N/A'):.4f} degrees")
        
    # Objects in field
    objects = results.get('objects_in_field', [])
    if objects:
        output.append("\n=== OBJECTS IN FIELD ===")
        for obj in objects[:10]:  # Limit to first 10 objects
            output.append(f"- {obj}")
        if len(objects) > 10:
            output.append(f"... and {len(objects) - 10} more objects")
    
    # Machine tags
    tags = results.get('machine_tags', [])
    if tags:
        output.append("\n=== MACHINE TAGS ===")
        for tag in tags:
            output.append(f"- {tag}")
    
    return "\n".join(output)

def main():
    """Main function to run the plate solver"""
    parser = argparse.ArgumentParser(description='Astrometry.net Plate Solver')
    parser.add_argument('image_path', help='Path to the astronomical image')
    parser.add_argument('--api-key', help='Astrometry.net API key')
    parser.add_argument('--center-ra', type=float, help='Approximate RA in degrees')
    parser.add_argument('--center-dec', type=float, help='Approximate Dec in degrees')
    parser.add_argument('--radius', type=float, help='Search radius in degrees')
    parser.add_argument('--output-dir', default='.', help='Directory to save results')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for plate solving')
    parser.add_argument('--interval', type=int, default=10, help='Interval in seconds between status checks')
    
    args = parser.parse_args()
    
    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get('ASTROMETRY_API_KEY')
    if not api_key:
        logger.error("Error: API key is required. Use --api-key or set the ASTROMETRY_API_KEY environment variable.")
        sys.exit(1)
    
    # Check if image exists
    if not os.path.isfile(args.image_path):
        logger.error(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize client and login
    client = AstrometryNetClient(api_key)
    client.login()
    
    # Upload image and get submission ID
    submission_id = client.upload_image(
        args.image_path,
        center_ra=args.center_ra,
        center_dec=args.center_dec,
        radius=args.radius
    )
    
    # Wait for processing to complete
    job_id = client.wait_for_results(submission_id, timeout=args.timeout, interval=args.interval)
    if not job_id:
        logger.error("Plate solving failed or timed out.")
        logger.info(f"You can check your submission status manually at: http://nova.astrometry.net/status/{submission_id}")
        sys.exit(1)
    
    # Get results
    results = client.get_results(job_id)
    print(format_results(results))
    
    # Download files
    client.download_wcs_file(job_id, args.output_dir)
    client.download_annotated_image(job_id, args.output_dir)
    
    # Save full JSON results
    output_path = os.path.join(args.output_dir, f"{job_id}_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Full results saved to {output_path}")

if __name__ == "__main__":
    main()