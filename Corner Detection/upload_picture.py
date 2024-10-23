import numpy as np
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO


# YOLOv8 Modell laden
model = YOLO('C:/Users/denni/PycharmProjects/TestProject/runs/detect/yolov8n_corner12/weights/best.pt')

# Funktion zur Erkennung der Eckpunkte A, B und C mit YOLO
def detect_yolo(image):
    results = model(image)
    points = {}

    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
        center_x = int((box[0] + box[2]) / 2)  # Mittelpunkt der Bounding Box (x)
        center_y = int((box[1] + box[3]) / 2)  # Mittelpunkt der Bounding Box (y)

        # Klassennamen im YOLO-Modell zuordnen
        if result.cls.item() == 0:  # A
            points["A"] = np.array([center_x, center_y])
        elif result.cls.item() == 1:  # B
            points["B"] = np.array([center_x, center_y])
        elif result.cls.item() == 2:  # C
            points["C"] = np.array([center_x, center_y])

    if len(points) != 3:
        raise ValueError("Nicht alle Eckpunkte (A, B, C) wurden erkannt!")

    return points

# Funktion zur Sortierung der Punkte: A oben links, B unten links, C unten rechts, D oben rechts
def sort_points(A, B, C, D):
    points = np.array([A, B, C, D])

    # Sortiere die Punkte nach der y-Koordinate (oben und unten)
    points = sorted(points, key=lambda x: x[1])

    # Sortiere die oberen Punkte nach der x-Koordinate
    top_points = sorted(points[:2], key=lambda x: x[0])  # Obere Punkte (A und D)
    # Sortiere die unteren Punkte nach der x-Koordinate
    bottom_points = sorted(points[2:], key=lambda x: x[0])  # Untere Punkte (B und C)

    # Weisen den sortierten Punkten die richtige Position zu
    A_sorted, D_sorted = top_points  # A ist oben links, D ist oben rechts
    B_sorted, C_sorted = bottom_points  # B ist unten links, C ist unten rechts

    return np.array([A_sorted, B_sorted, C_sorted, D_sorted], dtype=np.float32)

# Funktion zur Perspektivtransformation (Entzerren des Schachbretts)
def warp_perspective(image, src_points):
    dst_size = 800  # Zielgröße 800x800 Pixel für das quadratische Schachbrett
    dst_points = np.array([
        [0, 0],  # A' (oben links)
        [0, dst_size - 1],  # B' (unten links)
        [dst_size - 1, dst_size - 1],  # C' (unten rechts)
        [dst_size - 1, 0]  # D' (oben rechts)
    ], dtype=np.float32)

    # Perspektivtransformation berechnen
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Perspektivtransformation anwenden
    warped_image = cv2.warpPerspective(image, M, (dst_size, dst_size))

    return warped_image

# Funktion zum Drehen des Bildes um 90 Grad im Uhrzeigersinn
def rotate_image(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Funktion zur Visualisierung des 8x8-Rasters mit Koordinatensystem (von B)
def visualize_grid_with_coordinates(image, grid_size=8):
    step_size = image.shape[0] // grid_size  # Größe jeder Zelle
    image_copy = image.copy()

    # Zeichne horizontale Linien
    for i in range(1, grid_size):
        cv2.line(image_copy, (0, i * step_size), (image_copy.shape[1], i * step_size), (0, 0, 255), 2)

    # Zeichne vertikale Linien
    for i in range(1, grid_size):
        cv2.line(image_copy, (i * step_size, 0), (i * step_size, image_copy.shape[0]), (0, 0, 255), 2)

    # Füge Koordinaten (A-H und 1-8) hinzu, mit Bezugspunkt unten links (B)
    for i in range(grid_size):
        # Buchstaben A-H auf der X-Achse (von unten links beginnen)
        cv2.putText(image_copy, chr(65 + i), (i * step_size + step_size // 3, image_copy.shape[0] - step_size // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Zahlen 1-8 auf der Y-Achse (von unten nach oben)
        cv2.putText(image_copy, str(grid_size - i), (step_size // 4, (i + 1) * step_size - step_size // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Zeige das Bild mit Raster und Koordinaten
    plt.figure(figsize=(10, 10))
    plt.imshow(image_copy)
    plt.title("8x8 Raster mit Koordinatensystem (von unten links)")
    plt.axis('off')
    plt.show()

# Lade das Bild (Pfad anpassen)
image_path = 'C:/Test_Images_Yolo/IMG_5440.jpeg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# YOLO-Modell anwenden, um die Punkte A, B und C zu erkennen
detected_points = detect_yolo(image_rgb)

# Berechnung des Vektors BC und der Ecke D
A = detected_points["A"]
B = detected_points["B"]
C = detected_points["C"]
BC = C - B
D_calculated = A + BC

# Korrektur der Ecke D
correction_vector = np.array([324, 222])
D_corrected = D_calculated + correction_vector

# Sortiere die Punkte, um sicherzustellen, dass A oben links, B unten links, C unten rechts, D oben rechts ist
sorted_points = sort_points(A, B, C, D_corrected.astype(int))

# Perspektivtransformation: Entzerrung des Schachbretts
warped_image = warp_perspective(image_rgb, sorted_points)

# Drehe das entzerrte Bild um 90 Grad im Uhrzeigersinn
rotated_warped_image = rotate_image(warped_image)

# Zeige das gedrehte entzerrte Schachbrett
plt.figure(figsize=(10, 10))
plt.imshow(rotated_warped_image)
plt.title("Entzerrtes und gedrehtes Schachbrett")
plt.axis('off')
plt.show()

# Visualisiere das 8x8-Raster mit Koordinatensystem (von B)
visualize_grid_with_coordinates(rotated_warped_image)
