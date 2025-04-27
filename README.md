# 🔭 AstroPlate Solver

## Unlock the Secrets of the Night Sky

AstroPlate Solver is a powerful web application that helps astronomers and astrophotography enthusiasts identify the exact position and contents of astronomical images. By leveraging the Astrometry.net plate solving service, this tool can precisely determine the coordinates, field of view, and celestial objects captured in your images.

## Features

- **Automated Plate Solving**: Upload your astronomical images and get accurate astrometric calibration
- **Constellation Identification**: Learn which constellation your image belongs to
- **Object Recognition**: Identify stars, galaxies, nebulae, and other celestial objects in your field of view
- **Coordinate Display**: Get precise Right Ascension and Declination coordinates
- **Field of View Analysis**: Determine the exact scale and orientation of your image
- **Annotated Results**: View your image with stars and deep sky objects labeled
- **WCS File Export**: Download the World Coordinate System file for use in other astronomy software
- **Console Logging**: Track the plate solving process in real-time
- **Example Images**: Try the application with built-in example images

## Getting Started

### Prerequisites

- Python 3.7+
- Streamlit
- An API key from Astrometry.net (free to obtain)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/astroplate-solver.git
cd astroplate-solver
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

### Usage

1. Open the application in your web browser (typically at http://localhost:8501)
2. Enter your Astrometry.net API key in the sidebar
3. Upload an astronomical image or select one of the example images
4. Click the "Solve Plate" button and wait for the process to complete
5. Explore the results, including the annotated image, coordinates, and identified objects

## Advanced Options

- **Timeout Settings**: Adjust the maximum time allowed for plate solving
- **Console Logs**: View detailed logs of the plate solving process
- **Direct Astrometry.net Access**: Access your results directly on the Astrometry.net website

## How it Works

AstroPlate Solver follows these steps to analyze your astronomical images:

1. **Image Upload**: Your image is securely uploaded to the application
2. **API Submission**: The image is sent to the Astrometry.net service via their API
3. **Pattern Matching**: Astrometry.net matches star patterns in your image with its database
4. **Calibration**: Once a match is found, precise astrometric calibration is performed
5. **Results Processing**: The application processes and organizes the results
6. **Visualization**: The results are displayed in an intuitive interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Astrometry.net](http://astrometry.net/) for their incredible plate solving service
- [Astropy](https://www.astropy.org/) for astronomical calculations and FITS handling
- [Streamlit](https://streamlit.io/) for the web application framework

## Contact

- **Developer**: Sirjan Singh
- **GitHub**: [yourusername](https://github.com/SirjanSingh)
- **Email**: sirjan.singh036@gmail.com

---

Made with ❤️ for the astronomy community
