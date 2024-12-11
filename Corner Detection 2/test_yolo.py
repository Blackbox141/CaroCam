from ultralytics import YOLO

def train_yolo_model():
    # 1. Initialisiere das YOLOv8-Modell (Nano-Variante)
    model = YOLO("yolov8n.pt")  # Verwende das vortrainierte yolov8n-Modell

    # 2. Starte das Training mit dem heruntergeladenen Datensatz
    model.train(
        data="C:/Users/denni/PycharmProjects/TestProject/Other-Data-for-CaroCam-6/data.yaml",  # Pfad zur data.yaml im heruntergeladenen Datensatz
        epochs=50,  # Anzahl der Epochen, z.B. 50
        imgsz=1400,  # Bildgröße
        batch=16,  # Batch-Größe
        name='yolov8n_corner'  # Name des Trainierten Modells
    )

    # 3. Optional: Speichere das trainierte Modell
    model_path = "./runs/train/yolov8n_corner/weights/best.pt"
    print(f"Das Modell wurde erfolgreich trainiert und hier gespeichert: {model_path}")

if __name__ == '__main__':
    # Multiprocessing wird unter Windows korrekt initialisiert
    import multiprocessing
    multiprocessing.freeze_support()

    # Starte das Training
    train_yolo_model()

# C:/Test_Images_Yolo/IMG_5440.jpeg