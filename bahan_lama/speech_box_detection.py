import cv2
import numpy as np

def deteksi_bentuk(img, output_file):
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplikasikan threshold
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Cari kontur
    kontur, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variabel untuk menyimpan kontur kotak besar
    kotak_besar = None

    for kontur in kontur:
        # Hitung luas kontur
        luas = cv2.contourArea(kontur)

        # Filter kontur berdasarkan luas (kotak besar)
        if luas > 100000:  
            # Hitung perimeter
            peri = cv2.arcLength(kontur, True)

            # Aproximasi kontur
            approx = cv2.approxPolyDP(kontur, 0.01 * peri, True)

            # Deteksi kotak besar
            if len(approx) >= 4:  
                kotak_besar = kontur
                cv2.drawContours(img, [kontur], -1, (0, 255, 0), 10)

                # Buat bounding box
                x, y, w, h = cv2.boundingRect(kontur)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 10)

    # Cari kontur lingkaran/bulat/oval
    kontur, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for kontur in kontur:
        # Hitung luas kontur
        luas = cv2.contourArea(kontur)

        # Filter kontur berdasarkan luas (lingkaran/bulat/oval)
        if luas > 100:  
            # Hitung perimeter
            peri = cv2.arcLength(kontur, True)

            # Aproximasi kontur
            approx = cv2.approxPolyDP(kontur, 0.9 * peri, True)

            # Deteksi lingkaran/bulat/oval
            if len(approx) < 4:  
                # Periksa apakah kontur berada di dalam kotak besar
                if kotak_besar is not None:
                    x, y, w, h = cv2.boundingRect(kontur)
                    x1, y1, w1, h1 = cv2.boundingRect(kotak_besar)
                    if (x >= x1 and y >= y1 and x+w <= x1+w1 and y+h <= y1+h1):
                        cv2.drawContours(img, [kontur], -1, (0, 0, 255), 1)

    # Simpan hasil
    cv2.imwrite(output_file, img)

# Baca gambar
img = cv2.imread('./temp_images/page_2.jpg')

# Deteksi bentuk
deteksi_bentuk(img, 'hasil_deteksi.jpg')

print("Hasil deteksi disimpan sebagai hasil_deteksi.jpg")