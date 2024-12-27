# Manga Text Detector

A Python script for automatically detecting and extracting text bubbles from manga pages. This tool uses computer vision techniques to identify speech bubbles, extract their content, and perform OCR (Optical Character Recognition) on the text.

## Features

- Detects speech bubbles in manga pages
- Extracts text using Tesseract OCR
- Generates multiple output formats:
  - Annotated image with detected bubbles
  - Overlay of extracted bubbles
  - Transparent PNG with color-coded text
  - Individual bubble images
  - Text file with bubble positions and extracted text

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Tesseract OCR
- pytesseract

## Installation

1. Install Tesseract OCR:

   ```bash
   # Windows
   Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

   # Linux
   sudo apt install tesseract-ocr
   ```

2. Install Python dependencies:
   ```bash
   pip install opencv-python numpy pytesseract
   ```

## Usage

1. Place your manga page image in the `temp_images` folder
2. Run the script:
   ```python
   python auto_detect06.py
   ```

## Configuration

You can adjust the following parameters in the script:

- Minimum bubble area: `min_area = 500`
- Minimum bubble dimensions:
  ```python
  MIN_BUBBLE_WIDTH = 50
  MIN_BUBBLE_HEIGHT = 100
  ```
- Maximum bubble dimensions:
  ```python
  MAX_BUBBLE_WIDTH = 1200
  MAX_BUBBLE_HEIGHT = 1500
  ```

## Output Files

The script generates several output files:

1. `manga_auto_detection_01.jpg`: Original image with detected bubbles outlined
2. `extracted_overlay.jpg`: Extracted bubbles overlay
3. `extracted_bubbles_transparent.png`: Transparent PNG with color-coded text
4. `bubble_positions_with_text.txt`: Text file containing bubble positions and OCR results
5. `/extracted_bubbles/`: Folder containing individual bubble images

## Text Processing

The script processes text in the following ways:

- Converts image to grayscale
- Applies thresholding to identify white areas (potential bubbles)
- Uses morphological operations to clean up noise
- Performs OCR on detected regions

## Color Coding

The transparent output uses the following color scheme:

- White/gray areas become transparent
- Black text becomes red
- Other colors remain unchanged

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is open-source and available under the MIT License.
