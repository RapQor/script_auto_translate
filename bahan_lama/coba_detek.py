import cv2
import numpy as np

def deteksi_bentuk(img, output_file):
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplikasikan threshold
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Cari kontur
    kontur, hierarcy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, kontur in enumerate(kontur):
        if i == 0:
            continue
    
        epsilon = 0.01*cv2.arcLength(kontur, True)
        approx = cv2.approxPolyDP(kontur, epsilon, True)

        cv2.drawContours(img, kontur, 0, (0,0,0), 4)

        x, y, w, h = cv2.boundingRect(approx)
        x_mid = int(x + w / 3)
        y_mid = int(y + h / 1.5)

        coords = (x_mid, y_mid)
        colour = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if len(approx) < 6:
            cv2.drawContours(img, [kontur], -1, (0, 0, 255), 5)
        elif len(approx) >= 5:
            cv2.drawContours(img, [kontur], -1, (0, 255, 0), 5)
        # elif len(approx) >= 6:
        #     cv2.drawContours(img, [kontur], -1, (255, 0, 0), 5)
        # elif len(approx) >= 4:
        #     cv2.drawContours(img, [kontur], -1, (255, 255, 0), 5)
        # else:
        #     cv2.drawContours(img, [kontur], -1, (0, 255, 255), 5)

        # if len(approx) == 4:
        #     cv2.putText(img, 'INI SEGITIGA', coords, font, 1, colour, 2)
        # elif len(approx) == 5:
        #     cv2.putText(img, 'INI SEGILIMA', coords, font, 1, colour, 1)
        # elif len(approx) == 6:
        #     cv2.putText(img, 'INI SEGIENAM', coords, font, 1, colour, 1)
        # else:
        #     cv2.putText(img, 'INI LINGKARAN', coords, font, 1, colour, 1)


    # Simpan hasil
    cv2.imwrite(output_file, img)

# Baca gambar
img = cv2.imread('./temp_images/page_2.jpg')

# Deteksi bentuk
deteksi_bentuk(img, 'hasil_deteksi.jpg')

print("Hasil deteksi disimpan sebagai hasil_deteksi.jpg")