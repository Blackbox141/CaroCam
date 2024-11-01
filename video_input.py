import numpy as np
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import os
import requests
import chess
import chess.svg
import tempfile
import io
import webbrowser

# Holen des aktuellen Skriptverzeichnisses
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relativer Pfad zum Modell zur Erkennung der Schachfiguren
piece_model_path = os.path.join(BASE_DIR, 'Figure Detection', 'runs', 'detect', 'yolov8-chess28', 'weights', 'best.pt')

# Relativer Pfad zum Modell zur Erkennung der Eckpunkte
corner_model_path = os.path.join(BASE_DIR, 'Corner Detection', 'runs', 'detect', 'yolov8n_corner12', 'weights', 'best.pt')

# Laden der YOLO-Modelle mit den relativen Pfaden
if os.path.exists(piece_model_path) and os.path.exists(corner_model_path):
    piece_model = YOLO(piece_model_path)
    corner_model = YOLO(corner_model_path)
else:
    print(f"Modelldatei nicht gefunden. Überprüfe die Pfade:\nPiece model: {piece_model_path}\nCorner model: {corner_model_path}")
    exit()

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
PERCENT_AB = 0.17  # Beispielwert: 17% der Strecke AB
PERCENT_BC = -0.07  # Beispielwert: -7% der Strecke BC

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
        print("Nicht alle Eckpunkte (A, B, C) wurden erkannt!")
        return None, None

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

    return warped_image, M

def rotate_image(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def generate_fen_from_board(midpoints, labels, grid_size=8):
    # Erstelle ein leeres Schachbrett (8x8) in Form einer Liste
    board = [['' for _ in range(grid_size)] for _ in range(grid_size)]

    step_size = 800 // grid_size  # Größe jeder Zelle (entspricht der Größe des entzerrten Bildes)

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
            # Ignoriere Figuren außerhalb des Schachbretts
            print(f"Figur '{label}' an Position ({x:.2f}, {y:.2f}) ist außerhalb des Schachbretts und wird ignoriert.")

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

    # Verbinde alle Zeilen mit Schrägstrichen und hänge standardmäßige FEN-Informationen an
    fen_string = '/'.join(fen_rows) + " w - - 0 1"  # "w" (Weiß am Zug), keine Rochade, keine En-passant, 0 Züge, Zug 1

    return fen_string

def analyze_fen_with_stockfish(fen, depth=15):
    url = 'https://stockfish.online/api/s/v2.php'

    # URL mit den Parametern FEN und Tiefe aufbauen
    params = {
        'fen': fen,
        'depth': min(depth, 15)  # Tiefe darf nicht größer als 15 sein
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

def get_board_svg(fen, best_move):
    import chess
    import chess.svg

    board = chess.Board(fen)

    # Wenn ein bester Zug vorhanden ist, erstellen wir einen Pfeil
    if best_move:
        try:
            move = chess.Move.from_uci(best_move)
            arrows = [chess.svg.Arrow(move.from_square, move.to_square, color='#FF0000')]
        except ValueError:
            arrows = []
    else:
        arrows = []

    # Erzeuge ein SVG-Bild des Schachbretts mit dem Pfeil
    board_svg = chess.svg.board(board=board, size=400, arrows=arrows)
    return board_svg

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

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Überprüfen, ob das Video geladen werden konnte
    if not cap.isOpened():
        print("Fehler beim Laden des Videos.")
        return

    previous_fen = None
    move_counter = 0

    # Frame-Rate reduzieren (z.B. jedes 10. Frame verarbeiten)
    frame_skip = 10
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Ende des Videos

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Überspringe dieses Frame

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Schritt 1: Erkennung der Schachfiguren
            piece_midpoints, piece_labels, piece_results = detect_pieces(frame)

            # Schritt 2: Erkennung der Eckpunkte
            detected_points, corner_results = detect_corners(frame)
            if detected_points is None:
                print("Nicht alle Eckpunkte erkannt. Überspringe dieses Frame.")
                continue

            # Schritt 3: Berechnung der Ecke D und Perspektivtransformation
            A = detected_points["A"]
            B = detected_points["B"]
            C = detected_points["C"]
            D_calculated = calculate_point_D(A, B, C)

            # Anpassung des Punktes D mit dem dynamischen Korrekturvektor
            D_corrected = adjust_point_D(A, B, C, D_calculated)

            # Optional: Visualisierung von D_calculated und D_corrected
            # plot_points_with_correction(frame, A, B, C, D_calculated, D_corrected)

            # Sortiere die Punkte
            sorted_points = sort_points(A, B, C, D_corrected.astype(int))

            # Perspektivtransformation durchführen
            warped_image, M = warp_perspective(image_rgb, sorted_points)

            # Schritt 4: Schachfiguren transformieren
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

            # Schritt 5: Generiere die FEN-Notation
            fen_string = generate_fen_from_board(rotated_midpoints, piece_labels)

            # Vergleiche die FEN-Notation mit der vorherigen
            if fen_string != previous_fen:
                print(f"Neuer Zug erkannt: {fen_string}")
                previous_fen = fen_string

                # Schritt 6: Analyse der FEN-Notation mit der Stockfish API
                best_move = analyze_fen_with_stockfish(fen_string)

                # Schritt 7: Darstellung der FEN-Notation mit dem empfohlenen Zug
                board_svg = get_board_svg(fen_string, best_move)

                # Speichere das SVG-Bild in einer temporären Datei
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.svg') as f:
                    f.write(board_svg)
                    svg_filename = f.name

                # Öffne das SVG-Bild im Standard-Webbrowser
                webbrowser.open('file://' + os.path.realpath(svg_filename))

            # Zeige das aktuelle Frame an (optional)
            cv2.imshow('Aktuelles Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Fehler bei der Verarbeitung des Frames: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()
    print("Videoverarbeitung abgeschlossen.")

def main():
    print("Schachbrett- und Figuren-Erkennung aus Video")

    # Video-Pfad eingeben
    video_path = input("Gib den Pfad zum Schachspiel-Video ein: ")

    if os.path.exists(video_path):
        process_video(video_path)
    else:
        print("Videodatei nicht gefunden. Bitte überprüfe den Pfad.")

if __name__ == "__main__":
    main()
# Beispielvideopfad:
# C:/Test_Images_Yolo/IMG_6135.mp4
