import cv2
import numpy as np
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_and_extract_text_bubbles(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    output = image.copy()
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold untuk mendapatkan area putih (dialog bubbles)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    # Morphological operations untuk membersihkan noise
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Temukan kontur
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Parameter bubble
    min_area = 500
    MIN_BUBBLE_WIDTH = 50
    MIN_BUBBLE_HEIGHT = 100
    MAX_BUBBLE_WIDTH = 1200
    MAX_BUBBLE_HEIGHT = 1500 
    
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            
            if (w >= MIN_BUBBLE_WIDTH and h >= MIN_BUBBLE_HEIGHT and 
                w <= MAX_BUBBLE_WIDTH and h <= MAX_BUBBLE_HEIGHT):
                
                # Buat mask untuk area bubble
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, (255), -1)
                
                # Ekstrak area bubble
                bubble_region = cv2.bitwise_and(image, image, mask=mask)
                bubble_image = bubble_region[y:y+h, x:x+w]
                
                # Jalankan OCR untuk mengecek apakah terdapat teks
                gray_bubble = cv2.cvtColor(bubble_image, cv2.COLOR_BGR2GRAY)
                ocr_result = pytesseract.image_to_string(gray_bubble, lang='ind').strip()
                
                if ocr_result:
                    # Gambar kotak pada output
                    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.putText(output, f'bubble {idx}', (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Simpan hasil
    cv2.imwrite('manga_auto_detection_01.jpg', output)
    
    return {
        'detection_result': 'manga_auto_detection_01.jpg'
    }

if __name__ == "__main__":
    image_path = "./temp_images/page_12.jpg"
    result = detect_and_extract_text_bubbles(image_path)