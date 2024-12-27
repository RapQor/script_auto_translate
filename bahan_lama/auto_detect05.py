import cv2
import numpy as np
import os
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_and_extract_text_bubbles(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    output = image.copy()
    
    # Buat folder untuk menyimpan hasil ekstraksi
    output_folder = 'extracted_bubbles'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold untuk mendapatkan area putih (dialog bubbles)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    # Morphological operations untuk membersihkan noise
    kernel = np.ones((5, 5), np.uint8)
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
    
    # Dictionary untuk menyimpan teks per bubble
    bubble_texts = {}
    extracted_overlay = np.zeros_like(image)
    
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
                
                # Jalankan OCR
                gray_bubble = cv2.cvtColor(bubble_image, cv2.COLOR_BGR2GRAY)
                ocr_data = pytesseract.image_to_data(gray_bubble, lang='eng', output_type=Output.DICT)
                
                # Gabungkan kata-kata menjadi satu kalimat
                bubble_words = [ocr_data['text'][i].strip() for i in range(len(ocr_data['text'])) 
                                if int(ocr_data['conf'][i]) > 0 and ocr_data['text'][i].strip()]
                
                if bubble_words:
                    bubble_texts[idx] = ' '.join(bubble_words)
                    
                    # Simpan bubble image
                    bubble_filename = os.path.join(output_folder, f'bubble_{idx}.png')
                    cv2.imwrite(bubble_filename, bubble_image)
                    
                    # Tambahkan ke overlay
                    extracted_overlay[y:y+h, x:x+w] = bubble_image
                    
                    # Gambar kotak pada output
                    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(output, f'bubble {idx}', (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Simpan hasil gambar
    detection_result_path = 'combined.jpg'
    overlay_result_path = 'extracted_overlay_combined.jpg'
    cv2.imwrite(detection_result_path, output)
    cv2.imwrite(overlay_result_path, extracted_overlay)
    
    # Simpan teks ke file
    text_output_path = 'combined.txt'
    with open(text_output_path, 'w', encoding='utf-8') as f:
        for idx, text in sorted(bubble_texts.items()):
            f.write(f'bubble {idx}: "{text}"\n')
    
    return {
        'detection_result': detection_result_path,
        'overlay_result': overlay_result_path,
        'text_result': text_output_path,
        'bubble_texts': bubble_texts
    }

if __name__ == "__main__":
    image_path = "./temp_images/page_15.jpg"
    result = detect_and_extract_text_bubbles(image_path)
    print("Proses selesai!")
    print(f"Hasil deteksi tersimpan di: {result['detection_result']}")
    print(f"Overlay hasil ekstraksi tersimpan di: {result['overlay_result']}")
    print(f"Bubble texts tersimpan di: {result['text_result']}")
    print("\nTeks yang diekstrak per bubble:")
    for idx, text in result['bubble_texts'].items():
        print(f'bubble {idx} = "{text}"')
