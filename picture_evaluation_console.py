import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import os
import requests
import chess
import chess.svg
import webbrowser
import tempfile
import io

# Holen des aktuellen Skriptverzeichnisses - Testbild C:/Test_Images_Yolo/IMG_6661.jpeg
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relativer Pfad zum Modell zur Erkennung der Schachfiguren
piece_model_path = os.path.join(BASE_DIR, 'Figure Detection Model', 'runs', 'detect', 'yolov8-chess32', 'weights', 'best.pt')

# Relativer Pfad zum Modell zur Erkennung der Eckpunkte
corner_model_path = os.path.join(BASE_DIR, 'Corner Detection Model', 'runs', 'detect', 'yolov8n_corner14', 'weights', 'best.pt')

# Laden der YOLO-Modelle mit den relativen Pfaden
if os.path.exists(piece_model_path) and os.path.exists(corner_model_path):
    piece_model = YOLO(piece_model_path)
    corner_model = YOLO(corner_model_path)
else:
    raise FileNotFoundError(f"Modelldatei nicht gefunden. Überprüfe die Pfade:\nPiece model: {piece_model_path}\nCorner model: {corner_model_path}")

# FEN-Zeichen für Figuren
FEN_MAPPING = {
    'Black Pawn': 'p',
    'Black Bishop': 'b',
    'Black King': 'k',
    'Black Queen': 'q',
    'Black Rook': 'r',
    'Black Knight': 'n',
    'White Pawn': 'P',
    'White Bishop': 'B',
    'White King': 'K',
    'White Queen': 'Q',
    'White Rook': 'R',
    'White Knight': 'N'
}

# Parameter für die Korrektur (kannst du hier anpassen)
PERCENT_AB = 0.17  # Beispielwert: 15% der Strecke AB
PERCENT_BC = -0.07  # Beispielwert: 10% der Strecke BC

# Funktionen
def detect_pieces(image):
    results = piece_model.predict(image, conf=0.1, iou=0.3, imgsz=1400)
    result = results[0]

    midpoints = []
    labels = []
    boxes = result.boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        cls = int(box.cls.cpu().numpy()[0])
        label = result.names[cls]

        # Berechnung des Mittelpunkts der unteren Hälfte der Bounding Box
        mid_x = x1 + (x2 - x1) / 2
        mid_y = y1 + (y2 - y1) * 0.75  # Mitte der unteren Hälfte

        midpoints.append([mid_x, mid_y])
        labels.append(label)

    return np.array(midpoints), labels, result

def detect_corners(image):
    results = corner_model(image)
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

    return points, results

def calculate_point_D(A, B, C):
    BC = C - B
    D_calculated = A + BC
    return D_calculated

def adjust_point_D(A, B, C, D_calculated, percent_AB=PERCENT_AB, percent_BC=PERCENT_BC):
    # Berechnung der Vektoren AB und BC
    AB = B - A
    BC = C - B

    # Berechnung des Korrekturvektors
    correction_vector = percent_AB * AB + percent_BC * BC

    # Anwenden des Korrekturvektors auf D_calculated
    D_corrected = D_calculated + correction_vector

    return D_corrected

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

def warp_perspective(image, src_points):
    dst_size = 800  # Zielgrösse 800x800 Pixel für das quadratische Schachbrett
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

    return warped_image, M

def rotate_image(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def plot_corners(image, points):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for label, point in points.items():
        ax.plot(point[0], point[1], 'ro')  # Rote Punkte für Eckpunkte
        ax.text(point[0], point[1], label, fontsize=12, color='white',
                bbox=dict(facecolor='red', alpha=0.5))
    plt.title("Erkannte Eckpunkte des Schachbretts")
    plt.axis('off')
    return fig  # Rückgabe des Figure-Objekts

def plot_pieces(image, result):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        cls = int(box.cls.cpu().numpy()[0])
        label = result.names[cls]
        conf = box.conf.cpu().numpy()[0]

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f'{label} {conf:.2f}', color='white', fontsize=12,
                bbox=dict(facecolor='green', alpha=0.5))

    plt.title("Erkannte Schachfiguren mit Bounding Boxes")
    plt.axis('off')
    return fig  # Rückgabe des Figure-Objekts

def plot_transformed_pieces(image, midpoints, labels):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for point, label in zip(midpoints, labels):
        ax.plot(point[0], point[1], 'ro')  # Roter Punkt für die Figur
        ax.text(point[0], point[1], label, fontsize=12, color='white',
                bbox=dict(facecolor='blue', alpha=0.5))

    plt.title("Transformierte Schachfiguren auf dem entzerrten Schachbrett")
    plt.axis('off')
    return fig  # Rückgabe des Figure-Objekts

def plot_final_board(image, midpoints, labels):
    grid_size = 8
    step_size = image.shape[0] // grid_size  # Grösse jeder Zelle

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    # Zeichne das Raster
    for i in range(grid_size + 1):
        ax.axhline(i * step_size, color='black', linewidth=1)
        ax.axvline(i * step_size, color='black', linewidth=1)

    # Füge Koordinaten (A-H und 1-8) hinzu
    for i in range(grid_size):
        ax.text(i * step_size + step_size / 2, image.shape[0] - 10, chr(65 + i),
                fontsize=12, color='black', ha='center', va='center')
        ax.text(10, i * step_size + step_size / 2, str(grid_size - i),
                fontsize=12, color='black', ha='center', va='center')

    # Zeichne die Figuren in ihren entsprechenden Feldern
    for point, label in zip(midpoints, labels):
        x, y = point
        col = int(x // step_size)
        row = int(y // step_size)

        # Prüfe, ob die Position innerhalb der Grenzen liegt
        if 0 <= row < grid_size and 0 <= col < grid_size:
            square_x = col * step_size + step_size / 2
            square_y = row * step_size + step_size / 2

            ax.text(square_x, square_y, FEN_MAPPING[label], fontsize=20, color='red', ha='center', va='center')
        else:
            # Ignoriere Figuren ausserhalb des Schachbretts
            print(f"Figur '{label}' an Position ({x:.2f}, {y:.2f}) ist ausserhalb des Schachbretts und wird nicht dargestellt.")

    plt.title("Schachbrett mit Figurenpositionen")
    plt.axis('off')
    return fig  # Rückgabe des Figure-Objekts

def generate_fen_from_board(midpoints, labels, grid_size=8):
    # Erstelle ein leeres Schachbrett (8x8) in Form einer Liste
    board = [['' for _ in range(grid_size)] for _ in range(grid_size)]

    step_size = 800 // grid_size  # Grösse jeder Zelle (entspricht der Grösse des entzerrten Bildes)

    # Fülle das Board mit den Figuren
    for point, label in zip(midpoints, labels):
        x, y = point
        col = int(x // step_size)
        row = int(y // step_size)

        # Prüfe, ob die Position innerhalb der Grenzen liegt
        if 0 <= row < grid_size and 0 <= col < grid_size:
            fen_char = FEN_MAPPING.get(label, '')
            board[row][col] = fen_char
        else:
            # Ignoriere Figuren ausserhalb des Schachbretts
            print(f"Figur '{label}' an Position ({x:.2f}, {y:.2f}) ist ausserhalb des Schachbretts und wird ignoriert.")

    # Erstelle die FEN-Notation
    fen_rows = []
    for row in board:
        fen_row = ''
        empty_count = 0
        for square in row:
            if square == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += square
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    # Verbinde alle Zeilen mit Schrägstrichen und hänge standardmässige FEN-Informationen an
    fen_string = '/'.join(fen_rows) + " w - - 0 1"  # "w" (Weiss am Zug), keine Rochade, keine En-passant, 0 Züge, Zug 1

    return fen_string

def analyze_fen_with_stockfish(fen, depth=15):
    url = 'https://stockfish.online/api/s/v2.php'

    # URL mit den Parametern FEN und Tiefe aufbauen
    params = {
        'fen': fen,
        'depth': min(depth, 15)  # Tiefe darf nicht grösser als 15 sein
    }

    try:
        # Sende GET-Anfrage mit den Parametern
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()  # Antwort im JSON-Format analysieren
            print("Antwort von Stockfish API:")
            print(data)

            if data.get("success"):
                raw_best_move = data.get("bestmove", "None")
                evaluation = data.get("evaluation", "None")
                mate = data.get("mate", None)

                # Parsen des best_move Strings
                if raw_best_move != "None":
                    # Beispiel für raw_best_move: 'bestmove c3b4 ponder d7d6'
                    tokens = raw_best_move.split()
                    if len(tokens) >= 2:
                        best_move = tokens[1]  # Extrahiere den eigentlichen Zug
                    else:
                        best_move = None
                        print("Konnte den besten Zug nicht aus der API-Antwort extrahieren.")
                else:
                    best_move = None

                # Ausgabe der Ergebnisse
                print(f"\nStockfish Bewertung:")
                print(f"Beste Zugempfehlung: {best_move}")
                print(f"Bewertung: {evaluation}")

                if mate is not None:
                    print(f"Matt in: {mate} Zügen")

                return best_move  # Rückgabe des besten Zugs
            else:
                print("Fehler in der API-Antwort:", data.get("error", "Unbekannter Fehler"))
        else:
            print(f"Fehler bei der Kommunikation mit der Stockfish API. Status code: {response.status_code}")
            print(f"Antwort: {response.text}")

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

    return None  # Falls kein Zug empfohlen wurde

def plot_board_with_move(fen, best_move):
    import chess
    import chess.svg
    import webbrowser
    import tempfile
    import os

    board = chess.Board(fen)

    # Wenn ein bester Zug vorhanden ist, erstellen wir einen Pfeil
    if best_move:
        try:
            move = chess.Move.from_uci(best_move)
            print(f"Empfohlener Zug: '{best_move}' wird als Pfeil dargestellt.")
            arrows = [chess.svg.Arrow(move.from_square, move.to_square, color='#FF0000')]
        except ValueError:
            print(f"Der empfohlene Zug '{best_move}' ist ungültig.")
            arrows = []
    else:
        print("Kein Zug wurde empfohlen oder der Zug ist ungültig.")
        arrows = []

    # Erzeuge ein SVG-Bild des Schachbretts mit dem Pfeil
    board_svg = chess.svg.board(board=board, size=400, arrows=arrows)
    # Speichere das SVG-Bild in einer temporären Datei
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.svg') as f:
        f.write(board_svg)
        svg_filename = f.name

    # Öffne das SVG-Bild im Standard-Webbrowser
    webbrowser.open('file://' + os.path.realpath(svg_filename))

def plot_points_with_correction(image, A, B, C, D_calculated, D_corrected):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    colors = {'A': 'red', 'B': 'green', 'C': 'blue', 'D_calculated': 'yellow', 'D_corrected': 'magenta'}

    # Zeichne die Punkte A, B, C
    points = {'A': A, 'B': B, 'C': C}
    for label, point in points.items():
        ax.plot(point[0], point[1], 'o', color=colors[label])
        ax.text(point[0], point[1], label, fontsize=12, color='white',
                bbox=dict(facecolor=colors[label], alpha=0.5))

    # Zeichne D_calculated
    ax.plot(D_calculated[0], D_calculated[1], 'o', color=colors['D_calculated'])
    ax.text(D_calculated[0], D_calculated[1], 'D_calculated', fontsize=12, color='white',
            bbox=dict(facecolor=colors['D_calculated'], alpha=0.5))

    # Zeichne D_corrected
    ax.plot(D_corrected[0], D_corrected[1], 'o', color=colors['D_corrected'])
    ax.text(D_corrected[0], D_corrected[1], 'D_corrected', fontsize=12, color='white',
            bbox=dict(facecolor=colors['D_corrected'], alpha=0.5))

    # Zeichne den Korrekturvektor als Pfeil von D_calculated zu D_corrected
    ax.arrow(D_calculated[0], D_calculated[1],
             D_corrected[0] - D_calculated[0], D_corrected[1] - D_calculated[1],
             head_width=10, head_length=15, fc='magenta', ec='magenta', linestyle='--')

    plt.title("Eckpunkte mit Korrektur des Punktes D")
    plt.axis('off')
    plt.show()

def main():
    print("Schachbrett- und Figuren-Erkennung")

    # Bildpfad eingeben
    image_path = input("Gib den Pfad zum Schachbrettbild ein: ")

    if os.path.exists(image_path):
        # Bild lesen
        image = cv2.imread(image_path)
        if image is None:
            print("Fehler beim Laden des Bildes.")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Schritt 1: Erkennung der Schachfiguren und Visualisierung
        piece_midpoints, piece_labels, piece_results = detect_pieces(image)
        fig1 = plot_pieces(image, piece_results)
        plt.show()

        # Schritt 2: Erkennung der Eckpunkte und Visualisierung
        detected_points, corner_results = detect_corners(image)
        fig2 = plot_corners(image, detected_points)
        plt.show()

        # Schritt 3: Berechnung der Ecke D und Perspektivtransformation
        A = detected_points["A"]
        B = detected_points["B"]
        C = detected_points["C"]
        D_calculated = calculate_point_D(A, B, C)

        # Anpassung des Punktes D mit dem dynamischen Korrekturvektor
        D_corrected = adjust_point_D(A, B, C, D_calculated)

        # Visualisierung von D_calculated und D_corrected mit Korrekturvektor
        plot_points_with_correction(image, A, B, C, D_calculated, D_corrected)

        # Sortiere die Punkte
        sorted_points = sort_points(A, B, C, D_corrected.astype(int))

        # Perspektivtransformation durchführen
        warped_image, M = warp_perspective(image_rgb, sorted_points)
        fig3 = plt.figure(figsize=(8, 8))
        plt.imshow(warped_image)
        plt.title("Entzerrtes Schachbrett")
        plt.axis('off')
        plt.show()

        # Schritt 4: Schachfiguren transformieren und visualisieren
        ones = np.ones((piece_midpoints.shape[0], 1))
        piece_midpoints_homogeneous = np.hstack([piece_midpoints, ones])
        transformed_midpoints = M @ piece_midpoints_homogeneous.T
        transformed_midpoints /= transformed_midpoints[2, :]  # Homogenisierung
        transformed_midpoints = transformed_midpoints[:2, :].T  # Zurück zu kartesischen Koordinaten

        # Drehe das entzerrte Bild und die transformierten Mittelpunkte
        rotated_warped_image = rotate_image(warped_image)

        rotated_midpoints = np.zeros_like(transformed_midpoints)
        rotated_midpoints[:, 0] = rotated_warped_image.shape[1] - transformed_midpoints[:, 1]
        rotated_midpoints[:, 1] = transformed_midpoints[:, 0]

        fig4 = plot_transformed_pieces(rotated_warped_image, rotated_midpoints, piece_labels)
        plt.show()

        # Schritt 5: Generiere die FEN-Notation
        fen_string = generate_fen_from_board(rotated_midpoints, piece_labels)
        print(f"FEN-Notation: {fen_string}")

        # Schritt 6: Visuelle Darstellung der Figuren im Raster
        fig5 = plot_final_board(rotated_warped_image, rotated_midpoints, piece_labels)
        plt.show()

        # Schritt 7: Analyse der FEN-Notation mit der Stockfish API
        best_move = analyze_fen_with_stockfish(fen_string)

        # Schritt 8: Darstellung der FEN-Notation mit dem empfohlenen Zug
        plot_board_with_move(fen_string, best_move)

    else:
        print("Bilddatei nicht gefunden. Bitte überprüfe den Pfad.")

if __name__ == "__main__":
    main()
# Beispielbildpfade:
# C:/Test_Images_Yolo/IMG_zug1.jpeg
# C:/Test_Images_Yolo/screenshot_video_1.png
# C:/Test_Images_Yolo/IMG_6043_Pro_Max.jpg
# C:/Test_Images_Yolo/IMG_6043_komisch.jpg
# C:/Test_Images_Yolo/FA6BC7CD-4797-4807-A376-806540D92FE2_1_105_c.jpeg