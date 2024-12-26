from ultralytics import YOLO

def main():
    # Laden des trainierten Modells
    model = YOLO('runs/detect/yolov8-chess4/weights/best.pt')

    # Evaluierung des Modells auf dem Validierungsdatensatz
    val_results = model.val(
        data='data.yaml',  # Passe den Pfad zu deiner data.yaml an
        conf=0.001,  # Niedrige Konfidenzschwelle
        iou=0.7,  # NMS-IoU-Schwelle
        save_json=True,  # Speichert die Vorhersagen und Labels im COCO-Format
        plots=False,
        save=False
    )

    # Die Konfusionsmatrix wird automatisch erstellt und gespeichert
    # Du findest sie im Ordner 'runs/val/exp/confusion_matrix.png'

if __name__ == '__main__':
    main()
