import cv2
import time
import numpy as np
from scipy.signal import wiener

FPS = 144
frame_time = 1 / FPS

EDGES_PERCENTAGE = 1

# Ouvre la webcam (généralement l'index 0)
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
best_features = np.zeros_like(frame)

if not cap.isOpened():
    print("Impossible d'ouvrir la webcam")
    exit()

while True:
    time.sleep(frame_time)
    ret, frame = cap.read()
    if not ret:
        print("Impossible de lire l'image")
        break

    frame = cv2.flip(frame, 1)

    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)

    # Convertit l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Applique le filtre de Wiener
    # gray_wiener = wiener(gray)
    gray_wiener = gray

    # Applique le filtre Sobel classique pour détecter les contours
    edges = cv2.Sobel(gray_wiener, cv2.CV_64F, 1, 1, ksize=5)
    edges = cv2.convertScaleAbs(edges)

    # edges = cv2.Canny(gray, 20, 100)

    # # Seuillage des contours
    # _, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)


    # Applique une opération morphologique pour fermer les petits trous
    kernel = np.ones((3, 3), np.uint8)

    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Applique une opération morphologique pour réduire le bruit
    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Met les contours en rouge
    edges_red = np.zeros_like(frame)
    edges_red[:, :, 2] = edges  # Canal rouge

    

    edges_red = cv2.GaussianBlur(edges_red, (5, 5), 0)

    # Met tous les pixels >0 à 255, les 0 restent 0
    edges_red = cv2.morphologyEx(edges_red, cv2.MORPH_OPEN, kernel)

    # Calcule la moyenne des pixels > 0 dans edges_red
    nonzero_pixels = edges_red[edges_red > 0]
    if nonzero_pixels.size > 0:
        mean_nonzero = np.mean(nonzero_pixels)
    else:
        mean_nonzero = 0
    # print(f"Moyenne des pixels > 0 dans edges_red: {mean_nonzero}")
    illuminance = mean_nonzero

    edges_red = np.where(edges_red > (2 * illuminance), 255, 0).astype(np.uint8)

    
    edges_red = cv2.morphologyEx(edges_red, cv2.MORPH_OPEN, kernel)
    edges_red = cv2.morphologyEx(edges_red, cv2.MORPH_CLOSE, kernel)
    edges_red = cv2.morphologyEx(edges_red, cv2.MORPH_CLOSE, kernel)
    edges_red = cv2.morphologyEx(edges_red, cv2.MORPH_OPEN, kernel)

    # Superpose les contours sur l'image originale
    frame = cv2.addWeighted(frame, 1-EDGES_PERCENTAGE, edges_red, EDGES_PERCENTAGE, 0)


    cv2.imshow('Webcam', frame)

    # Quitte si on appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()