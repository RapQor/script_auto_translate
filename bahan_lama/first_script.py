import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from googletrans import Translator
from reportlab.pdfgen import canvas

# Path configurations
POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def pdf_to_images(pdf_path, output_folder='temp_images'):
    """Convert PDF to images and save them"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    image_paths = []
    for i, image in enumerate(images):
        image_path = f'{output_folder}/page_{i+1}.jpg'
        image.save(image_path, 'JPEG')
        image_paths.append(image_path)
    
    return image_paths

def detect_text_regions(image):
    """Enhanced text region detection including bold text"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create different thresholds to catch both normal and bold text
    regions = []
    
    # For normal text
    thresh1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
    # For bold text
    thresh2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    
    for threshold in [thresh1, thresh2]:
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Increased minimum area for better filtering
                x, y, w, h = cv2.boundingRect(contour)
                # Extract ROI to check if it contains text
                roi = gray[y:y+h, x:x+w]
                
                # Use Tesseract to check if region contains text
                text = pytesseract.image_to_string(roi, lang='jpn', config='--psm 6')
                if text.strip():
                    # Add padding proportional to region size
                    padding_x = int(w * 0.1)
                    padding_y = int(h * 0.1)
                    regions.append({
                        'bbox': (max(0, x-padding_x),
                                max(0, y-padding_y),
                                min(image.shape[1], x+w+padding_x),
                                min(image.shape[0], y+h+padding_y)),
                        'contour': contour,
                        'original_size': (w, h)  # Store original text size
                    })
    
    # Merge overlapping regions
    merged_regions = []
    regions.sort(key=lambda x: x['bbox'][0])  # Sort by x coordinate
    
    for region in regions:
        if not merged_regions:
            merged_regions.append(region)
            continue
            
        last = merged_regions[-1]
        # Check for overlap
        if (region['bbox'][0] <= last['bbox'][2] and
            region['bbox'][2] >= last['bbox'][0] and
            region['bbox'][1] <= last['bbox'][3] and
            region['bbox'][3] >= last['bbox'][1]):
            # Merge regions
            x1 = min(last['bbox'][0], region['bbox'][0])
            y1 = min(last['bbox'][1], region['bbox'][1])
            x2 = max(last['bbox'][2], region['bbox'][2])
            y2 = max(last['bbox'][3], region['bbox'][3])
            merged_regions[-1]['bbox'] = (x1, y1, x2, y2)
            merged_regions[-1]['original_size'] = (x2-x1, y2-y1)
        else:
            merged_regions.append(region)
    
    return merged_regions

def estimate_original_font_size(region, text):
    """Estimate original font size based on region size and text length"""
    w, h = region['original_size']
    text_length = len(text)
    if text_length == 0:
        return 20  # Default size
    
    # Estimate based on height and character count
    estimated_size = int(h / (1 + text_length / 15))  # Adjust divisor as needed
    return min(max(estimated_size, 12), 50)  # Keep size within reasonable bounds

def translate_text(text, target_lang='en'):
    """Translate text with better error handling"""
    if not text.strip():
        return ""
    
    translator = Translator()
    try:
        translation = translator.translate(text.strip(), dest=target_lang)
        return translation.text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def overlay_text(image, text, region):
    """Improved text overlay with size matching"""
    # Convert to PIL for text handling
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Get region dimensions
    x1, y1, x2, y2 = region['bbox']
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Estimate initial font size
    initial_font_size = estimate_original_font_size(region, text)
    font_size = initial_font_size
    font = ImageFont.truetype("arial.ttf", font_size)
    
    # Adjust font size to fit
    text_wrapped = []
    while font_size > 8:
        text_wrapped = []
        current_line = []
        words = text.split()
        
        for word in words:
            test_line = current_line + [word]
            test_text = ' '.join(test_line)
            bbox = draw.textbbox((0, 0), test_text, font=font)
            if bbox[2] - bbox[0] <= box_width:
                current_line = test_line
            else:
                if current_line:
                    text_wrapped.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            text_wrapped.append(' '.join(current_line))
        
        # Check if text fits height
        total_height = len(text_wrapped) * (font_size + 2)
        if total_height <= box_height:
            break
        
        font_size -= 2
        font = ImageFont.truetype("arial.ttf", font_size)
    
    # Create semi-transparent background
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Draw text
    y = y1 + (box_height - total_height) // 2
    for line in text_wrapped:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = x1 + (box_width - text_width) // 2
        
        # Draw semi-transparent background for this line
        line_height = font_size + 2
        overlay_draw.rectangle([x-2, y-2, x+text_width+2, y+line_height],
                             fill=(255, 255, 255, 180))
        draw.text((x, y), line, font=font, fill=(0, 0, 0))
        y += line_height
    
    # Combine images
    img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_image_with_translation(image_path, target_lang='en'):
    """Process image with improved text detection and translation"""
    try:
        # Read image
        img = cv2.imread(image_path)
        original_img = img.copy()
        
        # Detect text regions
        text_regions = detect_text_regions(img)
        
        # Process each region
        for region in text_regions:
            x1, y1, x2, y2 = region['bbox']
            roi = original_img[y1:y2, x1:x2]
            
            # Get text from region with different configs for better detection
            text = pytesseract.image_to_string(roi, lang='jpn',
                                             config='--psm 6 --oem 3')
            if not text.strip():
                # Try alternative config for bold text
                text = pytesseract.image_to_string(roi, lang='jpn',
                                                 config='--psm 6 --oem 3 -c tessedit_char_blacklist="|"')
            
            if text.strip():
                translated_text = translate_text(text, target_lang)
                if translated_text:
                    img = overlay_text(img, translated_text, region)
        
        return img
        
    except Exception as e:
        print(f"Error in image processing: {str(e)}")
        raise

def images_to_pdf(image_paths, output_pdf):
    """Convert images to PDF with quality preservation"""
    c = canvas.Canvas(output_pdf)
    for image_path in image_paths:
        img = Image.open(image_path)
        img_width, img_height = img.size
        c.setPageSize((img_width, img_height))
        c.drawImage(image_path, 0, 0, img_width, img_height)
        c.showPage()
    c.save()

def translate_manga(input_pdf, output_pdf, target_lang='en'):
    """Main translation function"""
    try:
        # Verify paths
        if not os.path.exists(POPPLER_PATH):
            raise Exception(f"Poppler path not found: {POPPLER_PATH}")
        if not os.path.exists(TESSERACT_PATH):
            raise Exception(f"Tesseract executable not found: {TESSERACT_PATH}")
        
        # Convert PDF to images
        print("Converting PDF to images...")
        temp_folder = 'temp_images'
        image_paths = pdf_to_images(input_pdf, temp_folder)
        
        # Process each image
        print("Processing images and translating text...")
        translated_image_paths = []
        for i, image_path in enumerate(image_paths):
            print(f"Processing page {i+1}...")
            translated_img = process_image_with_translation(image_path, target_lang)
            translated_path = f'{temp_folder}/translated_page_{i+1}.jpg'
            cv2.imwrite(translated_path, translated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            translated_image_paths.append(translated_path)
        
        # Convert to PDF
        print("Converting processed images to PDF...")
        images_to_pdf(translated_image_paths, output_pdf)
        
        # Cleanup
        print("Cleaning up temporary files...")
        for path in image_paths + translated_image_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(temp_folder):
            os.rmdir(temp_folder)
        
        print("Translation completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    input_pdf = "manga_1.pdf"
    output_pdf = "translated_manga.pdf"
    target_lang = "en"
    
    translate_manga(input_pdf, output_pdf, target_lang)