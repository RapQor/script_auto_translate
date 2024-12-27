import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import sys

# Path configurations
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def check_tesseract():
    """Check if Tesseract is properly installed and configured"""
    if not os.path.exists(TESSERACT_PATH):
        print("Error: Tesseract is not installed or the path is incorrect.")
        print("Please install Tesseract-OCR and update the TESSERACT_PATH variable.")
        print("You can download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)

def detect_and_color_text(input_image_path, output_dir):
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Read the image
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error: Could not read image at {input_image_path}")
            return
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create masks for different colors
        red_mask = np.zeros_like(image)
        black_mask = np.zeros_like(image)
        
        # Process each contour
        for i, contour in enumerate(contours):
            # Filter small contours
            if cv2.contourArea(contour) < 100:
                continue
                
            # Create mask for current contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, (255), -1)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract text region
            roi = gray[y:y+h, x:x+w]
            
            try:
                # OCR to detect if region contains text
                text = pytesseract.image_to_string(roi, lang='ind')
                
                if len(text.strip()) > 0:
                    # Draw red version
                    cv2.drawContours(red_mask, [contour], -1, (0, 0, 255), -1)
                    
                    # Draw black version (enhanced)
                    cv2.drawContours(black_mask, [contour], -1, (0, 0, 0), -1)
            except pytesseract.TesseractError as e:
                print(f"Tesseract error processing ROI: {e}")
                continue
        
        # Create output images
        red_result = cv2.addWeighted(image, 1, red_mask, 0.5, 0)
        black_result = cv2.addWeighted(image, 1, black_mask, 0.5, 0)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_red.png'), red_result)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_black.png'), black_result)
        
        print(f"Successfully processed {input_image_path}")
        
    except Exception as e:
        print(f"Error processing {input_image_path}: {str(e)}")

def process_manga_pages(input_dir, output_dir):
    """Process all manga pages in a directory"""
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    files_processed = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            detect_and_color_text(input_path, output_dir)
            files_processed += 1
    
    print(f"\nProcessing complete. {files_processed} files processed.")

# Example usage
if __name__ == "__main__":
    # Check Tesseract installation before processing
    check_tesseract()
    
    input_directory = "manga_pages"
    output_directory = "processed_pages"
    
    print("Starting manga text detection...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    process_manga_pages(input_directory, output_directory)