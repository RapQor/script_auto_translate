import cv2
import numpy as np

def detect_text_bubbles(image_path):
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
    
    # Filter dan klasifikasi kontur
    bubbles = []
    titles = []
    side_text = []
    
    min_area = 500  # Minimum area threshold
    
    # Batasan ukuran untuk bubble
    MIN_BUBBLE_WIDTH = 50
    MIN_BUBBLE_HEIGHT = 100

    MAX_BUBBLE_WIDTH = 1200
    MAX_BUBBLE_HEIGHT = 1500
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            # Tambahkan pengecekan ukuran minimum untuk bubble
            if w >= MIN_BUBBLE_WIDTH and h >= MIN_BUBBLE_HEIGHT and w <= MAX_BUBBLE_WIDTH and h <= MAX_BUBBLE_HEIGHT:
                bubbles.append((x, y, w, h))
                
    # Dialog bubbles (Merah)
    for box in bubbles:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(output, f'bubble {w}x{h}', (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Simpan hasil
    output_path = 'manga_auto_detection.jpg'
    cv2.imwrite(output_path, output)
    return output_path

if __name__ == "__main__":
    image_path = "./temp_images/page_10.jpg"  # Ganti dengan path gambar Anda
    result_path = detect_text_bubbles(image_path)
    print(f"Hasil deteksi disimpan ke: {result_path}")