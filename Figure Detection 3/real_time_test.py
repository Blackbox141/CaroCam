#Mit diesem Skript kann das Modell in Echtzeit mit einer Webcam getestet werden.

from ultralytics import YOLO
import cv2

# Laden des trainierten Modells
model = YOLO('runs/detect/yolov8-chess/weights/best.pt')

# Videoquelle (z. B. Webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Vorhersage auf dem aktuellen Frame
    results = model.predict(source=frame, conf=0.5)

    # Ergebnisse auf dem Frame zeichnen
    annotated_frame = results[0].plot()

    # Frame anzeigen
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Beenden mit der Taste 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
