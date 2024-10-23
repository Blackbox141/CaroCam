import os

labels_dir = 'C:/Users/denni/PycharmProjects/pythonProject/data.yaml'  # Passe den Pfad an
invalid_classes = []
valid_range = range(0, 12)  # 0-11

for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                if class_id not in valid_range:
                    invalid_classes.append((label_file, class_id))

if invalid_classes:
    print("Gefundene ungültige Klassen:")
    for file, cls in invalid_classes:
        print(f"Datei: {file}, Klasse: {cls}")
else:
    print("Keine ungültigen Klassen gefunden.")
