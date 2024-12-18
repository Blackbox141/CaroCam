import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ultralytics import YOLO
import torch
import torchvision

# Laden des trainierten Modells
model = YOLO('runs/detect/yolov8-chess3/weights/best.pt')

# Funktion zur Anwendung von Non-Maximum Suppression (NMS)
def apply_nms(result, iou_threshold=0.5):
    boxes = result.boxes  # Boxes-Objekt

    # NMS anwenden
    keep = torchvision.ops.nms(boxes.xyxy, boxes.conf, iou_threshold)

    # Gefilterte Boxes zuweisen
    result.boxes = boxes[keep]

    return result

# Funktion, um die Ergebnisse mit Bounding Boxes zu plotten
def plot_results_with_custom_font(result, img, font_size=8, line_thickness=1):
    # Konvertiere das Bild von BGR zu RGB (falls noch nicht geschehen)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Größeres Ausgabe-Canvas
    fig, ax = plt.subplots(1, figsize=(12, 12))  # Canvas vergrößern (12x12 statt Standardgröße)
    ax.imshow(img_rgb)  # Verwende das Bild im RGB-Format

    # Definiere Farben für Schwarz und Weiß
    colors = {
        'Black Pawn': 'green',
        'Black Bishop': 'red',
        'Black King': 'blue',
        'Black Queen': 'purple',
        'Black Rook': 'orange',
        'Black Knight': 'yellow',
        'White Pawn': 'cyan',
        'White Bishop': 'magenta',
        'White King': 'lime',
        'White Queen': 'pink',
        'White Rook': 'teal',
        'White Knight': 'gold'
    }

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        width = x2 - x1
        height = y2 - y1

        x1, y1 = max(0, x1), max(0, y1)
        width, height = max(0, width), max(0, height)

        cls = int(box.cls.cpu().numpy()[0])
        conf = box.conf.cpu().numpy()[0]
        label = result.names[cls]
        conf_text = f"{label} {conf:.2f}"

        # Wähle die richtige Farbe basierend auf dem Label
        color = colors.get(label, 'white')  # Fallback auf 'white', falls nicht gefunden

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=line_thickness,  # Dünnere Linien für die Boxen
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Platzierung des Labels über oder unter der Box
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        ax.text(
            x1, label_y, conf_text,
            fontsize=font_size,  # Kleinere Schriftgröße
            color='white',
            bbox=dict(facecolor=color, alpha=0.5)
        )

    plt.axis('off')

    # Zeichnen des Canvas
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    # RGBA-Puffer abrufen und in ein NumPy-Array konvertieren
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(height, width, 4)

    # Konvertiere RGBA zu RGB
    annotated_img = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)

    plt.close(fig)
    return annotated_img

# Bild laden und vorhersagen
def predict_from_path(image_path):
    # Bild von Pfad laden
    img = cv2.imread(image_path)

    # Sicherstellen, dass das Bild korrekt geladen wurde
    if img is None:
        print(f"Fehler beim Laden des Bildes: {image_path}")
        return None

    # Modellvorhersage mit angepassten Schwellenwerten
    results = model.predict(img, conf=0.2, iou=0.3, imgsz=1400)
    result = results[0]

    # NMS anwenden
    result = apply_nms(result, iou_threshold=0.5)

    # Annotiertes Bild erstellen
    annotated_img = plot_results_with_custom_font(result, img, font_size=8, line_thickness=1)

    # Speichern des annotierten Bildes
    output_path = "annotated_output.png"
    cv2.imwrite(output_path, annotated_img)
    print(f"Das annotierte Bild wurde unter {output_path} gespeichert.")

    # Optional: Ausgabe des Bildes anzeigen
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Beispiel: Bild vom Pfad laden und vorhersagen
if __name__ == "__main__":
    # Manuell den Bildpfad eingeben
    image_path = input("Gib den Pfad zum Bild ein: ")

    # Vorhersage ausführen
    predict_from_path(image_path)

# C:/Test_Images_Yolo/IMG_6043_komisch.jpg