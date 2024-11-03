
# Update 03.11.24 - Code mit Korrekturvektor

import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import os
import requests
import chess
import chess.svg
import tempfile
import io
from PIL import Image

# Holen des aktuellen Skriptverzeichnisses
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relativer Pfad zum Modell zur Erkennung der Schachfiguren
piece_model_path = os.path.join(BASE_DIR, 'Figure Detection', 'runs', 'detect', 'yolov8-chess28', 'weights', 'best.pt')

# Relativer Pfad zum Modell zur Erkennung der Eckpunkte
corner_model_path = os.path.join(BASE_DIR, 'Corner Detection', 'runs', 'detect', 'yolov8n_corner12', 'weights', 'best.pt')

# Relativer Pfad zum Modell zur Erkennung des Spielers am Zug (Schachuhr)
clock_model_path = os.path.join(BASE_DIR, 'Clock Detection 1', 'runs', 'detect', 'yolov8-chess3', 'weights', 'best.pt')

# Laden der YOLO-Modelle mit den relativen Pfaden
if os.path.exists(piece_model_path) and os.path.exists(corner_model_path) and os.path.exists(clock_model_path):
    piece_model = YOLO(piece_model_path)
    corner_model = YOLO(corner_model_path)
    clock_model = YOLO(clock_model_path)
else:
    st.error(f"Modelldatei nicht gefunden. Überprüfe die Pfade:\nPiece model: {piece_model_path}\nCorner model: {corner_model_path}\nClock model: {clock_model_path}")
    st.stop()

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
        st.error("Nicht alle Eckpunkte (A, B, C) wurden erkannt!")
        st.stop()

    return points, results

def detect_player_turn(image):
    results = clock_model.predict(image, conf=0.1, iou=0.3, imgsz=1400)
    result = results[0]
    boxes = result.boxes

    # Initialisiere die Label-Liste
    labels = []

    for box in boxes:
        cls = int(box.cls.cpu().numpy()[0])
        label = result.names[cls]
        labels.append(label)

    # Bestimme anhand der Labels, wer am Zug ist
    if 'left' in labels:
        player_turn = 'left'
    elif 'right' in labels:
        player_turn = 'right'
    else:
        player_turn = None  # Fordere Benutzereingabe

    return player_turn, result

def calculate_point_D(A, B, C):
    BC = C - B
    D_calculated = A + BC
    return D_calculated

def adjust_point_D(A, B, C, D_calculated, percent_AB, percent_BC):
    # Berechnung der Vektoren AB und BC
    AB = B - A
    BC = C - B

    # Berechnung des Korrekturvektors
    correction_vector = percent_AB * AB + percent_BC * BC

    # Anwenden des Korrekturvektors auf D_calculated
    D_corrected = D_calculated + correction_vector

    return D_corrected

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
    return fig  # Rückgabe des Figure-Objekts

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

def plot_clock_detections(image, result):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        cls = int(box.cls.cpu().numpy()[0])
        label = result.names[cls]
        conf = box.conf.cpu().numpy()[0]

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f'{label} {conf:.2f}', color='white', fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5))

    plt.title("Erkannte Schachuhr Labels")
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
    step_size = image.shape[0] // grid_size  # Größe jeder Zelle

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
            # Ignoriere Figuren außerhalb des Schachbretts
            st.write(f"Figur '{label}' an Position ({x:.2f}, {y:.2f}) ist außerhalb des Schachbretts und wird nicht dargestellt.")

    plt.title("Schachbrett mit Figurenpositionen")
    plt.axis('off')
    return fig  # Rückgabe des Figure-Objekts

def generate_fen_from_board(midpoints, labels, grid_size=8, player_to_move='w'):
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
            st.write(f"Figur '{label}' an Position ({x:.2f}, {y:.2f}) ist außerhalb des Schachbretts und wird ignoriert.")

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

    # Verbinde alle Zeilen mit Schrägstrichen und hänge die Informationen an
    fen_string = '/'.join(fen_rows) + f" {player_to_move} - - 0 1"

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
            st.write("Antwort von Stockfish API:")
            st.write(data)

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
                        st.write("Konnte den besten Zug nicht aus der API-Antwort extrahieren.")
                else:
                    best_move = None

                # Ausgabe der Ergebnisse
                st.write(f"\n**Stockfish Bewertung:**")
                st.write(f"**Beste Zugempfehlung:** {best_move}")
                st.write(f"**Bewertung:** {evaluation}")

                if mate is not None:
                    st.write(f"**Matt in:** {mate} Zügen")

                return best_move  # Rückgabe des besten Zugs
            else:
                st.error("Fehler in der API-Antwort:", data.get("error", "Unbekannter Fehler"))
        else:
            st.error(f"Fehler bei der Kommunikation mit der Stockfish API. Status code: {response.status_code}")
            st.error(f"Antwort: {response.text}")

    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {e}")

    return None  # Falls kein Zug empfohlen wurde

def plot_board_with_move(fen, best_move):
    import chess
    import chess.svg

    board = chess.Board(fen)

    # Wenn ein bester Zug vorhanden ist, erstellen wir einen Pfeil
    if best_move:
        try:
            move = chess.Move.from_uci(best_move)
            st.write(f"Empfohlener Zug: '{best_move}' wird als Pfeil dargestellt.")
            arrows = [chess.svg.Arrow(move.from_square, move.to_square, color='#FF0000')]
        except ValueError:
            st.write(f"Der empfohlene Zug '{best_move}' ist ungültig.")
            arrows = []
    else:
        st.write("Kein Zug wurde empfohlen oder der Zug ist ungültig.")
        arrows = []

    # Erzeuge ein SVG-Bild des Schachbretts mit dem Pfeil
    board_svg = chess.svg.board(board=board, size=400, arrows=arrows)

    # Anzeige des SVG-Bildes in Streamlit
    st.components.v1.html(board_svg, height=500)

def main():
    st.title("Schachbrett- und Figuren-Erkennung")

    # Bild hochladen
    uploaded_file = st.file_uploader("Lade ein Bild des Schachbretts hoch", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Bild lesen
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        if image is None:
            st.error("Fehler beim Laden des Bildes.")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Schritt 1: Erkennung der Schachfiguren und Visualisierung
        piece_midpoints, piece_labels, piece_results = detect_pieces(image)
        # Wir zeigen diese Ergebnisse später an

        # Schritt 2: Erkennung der Eckpunkte und Visualisierung
        detected_points, corner_results = detect_corners(image)
        # Wir zeigen diese Ergebnisse später an

        # Schritt 3: Berechnung der Ecke D und Perspektivtransformation
        A = detected_points["A"]
        B = detected_points["B"]
        C = detected_points["C"]
        D_calculated = calculate_point_D(A, B, C)

        # Anpassung des Punktes D mit dem dynamischen Korrekturvektor
        st.write("Anpassung des Punktes D mit dem dynamischen Korrekturvektor")
        PERCENT_AB = st.slider("Prozentualer Anteil von AB (PERCENT_AB)", min_value=-1.0, max_value=1.0, value=0.17, step=0.01)
        PERCENT_BC = st.slider("Prozentualer Anteil von BC (PERCENT_BC)", min_value=-1.0, max_value=1.0, value=-0.07, step=0.01)

        D_corrected = adjust_point_D(A, B, C, D_calculated, PERCENT_AB, PERCENT_BC)

        # Sortiere die Punkte
        sorted_points = sort_points(A, B, C, D_corrected.astype(int))

        # Perspektivtransformation durchführen
        warped_image, M = warp_perspective(image_rgb, sorted_points)
        # Wir zeigen diese Ergebnisse später an

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

        # Schritt 4b: Erkennung des Spielers am Zug
        player_turn, clock_result = detect_player_turn(image)

        # Wenn der Spieler nicht erkannt wurde, fordere Benutzereingabe
        if player_turn is None:
            st.write("Konnte nicht erkennen, wer am Zug ist.")
            player_input = st.selectbox("Bitte wählen Sie, wer am Zug ist:", ("Weiß", "Schwarz"))
            if player_input == "Weiß":
                player_turn = 'white'
            else:
                player_turn = 'black'
        else:
            st.write(f"Erkannter Spieler am Zug: {player_turn}")

        # Mappe 'left' und 'right' zu 'w' und 'b' für die FEN-Notation
        if player_turn == 'left':
            player_to_move = 'b'  # Angenommen, 'left' entspricht Schwarz
        elif player_turn == 'right':
            player_to_move = 'w'  # Angenommen, 'right' entspricht Weiss
        elif player_turn == 'white':
            player_to_move = 'w'
        elif player_turn == 'black':
            player_to_move = 'b'

        # Schritt 5: Generiere die FEN-Notation
        fen_string = generate_fen_from_board(rotated_midpoints, piece_labels, player_to_move=player_to_move)
        st.write(f"**FEN-Notation:** {fen_string}")

        # Schritt 6: Analyse der FEN-Notation mit der Stockfish API
        best_move = analyze_fen_with_stockfish(fen_string)

        # Schritt 7: Darstellung der FEN-Notation mit dem empfohlenen Zug
        st.subheader("Empfohlenes Schachbrett mit Zugempfehlung")
        plot_board_with_move(fen_string, best_move)

        # Optionale Anzeige der Zwischenschritte
        if st.button("Zwischenschritte anzeigen"):
            st.subheader("Erkannte Schachfiguren mit Bounding Boxes")
            fig1 = plot_pieces(image, piece_results)
            st.pyplot(fig1)

            st.subheader("Erkannte Eckpunkte des Schachbretts")
            fig2 = plot_corners(image, detected_points)
            st.pyplot(fig2)

            st.subheader("Visualisierung von D_calculated und D_corrected mit Korrekturvektor")
            fig_correction = plot_points_with_correction(image, A, B, C, D_calculated, D_corrected)
            st.pyplot(fig_correction)

            st.subheader("Entzerrtes Schachbrett")
            fig3 = plt.figure(figsize=(8, 8))
            plt.imshow(warped_image)
            plt.title("Entzerrtes Schachbrett")
            plt.axis('off')
            st.pyplot(fig3)

            st.subheader("Transformierte Schachfiguren auf dem entzerrten Schachbrett")
            fig4 = plot_transformed_pieces(rotated_warped_image, rotated_midpoints, piece_labels)
            st.pyplot(fig4)

            st.subheader("Schachbrett mit Figurenpositionen")
            fig5 = plot_final_board(rotated_warped_image, rotated_midpoints, piece_labels)
            st.pyplot(fig5)

            st.subheader("Erkannte Schachuhr Labels")
            fig_clock = plot_clock_detections(image, clock_result)
            st.pyplot(fig_clock)

    else:
        st.write("Bitte lade ein Bild des Schachbretts hoch.")

if __name__ == "__main__":
    main()
