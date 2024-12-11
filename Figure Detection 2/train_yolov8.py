from ultralytics import YOLO

def main():
    # Pfad zur data.yaml-Datei
    DATA_CONFIG_PATH = 'C:/Users/denni/PycharmProjects/pythonProject/CaroCam-Project-7/data.yaml'

    # Initialisiere das Modell
    model = YOLO('yolov8n.pt')

    # Starte das Training
    model.train(
        data=DATA_CONFIG_PATH,
        epochs=70,
        imgsz=1400,
        batch=16,
        name='yolov8-chess',
        device=0
    )

if __name__ == '__main__':
    main()
