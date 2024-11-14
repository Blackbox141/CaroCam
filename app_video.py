import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import os
import requests
import chess
import chess.pgn
import tempfile
from PIL import Image

# Holen des aktuellen Skriptverzeichnisses
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relativer Pfad zu den Modellen
piece_model_path = os.path.join(BASE_DIR, 'Figure Detection', 'runs', 'detect', 'yolov8-chess28', 'weights', 'best.pt')
corner_model_path = os.path.join(BASE_DIR, 'Corner Detection', 'runs', 'detect', 'yolov8n_corner12', 'weights', 'best.pt')
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

# Festlegen der festen Werte für den Korrekturvektor
PERCENT_AB = 0.17  # Beispielwert: 17% der Strecke AB
PERCENT_BC = -0.07  # Beispielwert: -7% der Strecke BC

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
    elif 'hold' in labels:
        player_turn = 'hold'
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

def generate_fen_from_board(midpoints, labels, grid_size=8, player_to_move='w'):
    # Erstelle ein leeres Schachbrett (8x8) als Liste von Listen
    board = [['' for _ in range(grid_size)] for _ in range(grid_size)]

    step_size = 800 // grid_size  # Größe jeder Zelle (entspricht der Größe des entzerrten Bildes)

    # Fülle das Board mit den Figuren basierend auf den erkannten Mittelpunkten
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
            pass

    # Reihenfolge der Reihen umkehren, um die FEN-Notation korrekt zu erstellen
    board = board[::-1]

    # Spalten in jeder Reihe umkehren, um die Spiegelung zu beheben
    for row in board:
        row.reverse()

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

    # Verbinde alle Zeilen mit Schrägstrichen und füge die zusätzlichen Informationen hinzu
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

def plot_board_with_move(fen, best_move, white_side):
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

    # Setze das Brett auf die entsprechende Perspektive
    if white_side == "Links":
        flipped = False  # Brett aus der Sicht von Weiß
    else:
        flipped = True  # Brett aus der Sicht von Schwarz

    # Erzeuge ein SVG-Bild des Schachbretts mit dem Pfeil
    board_svg = chess.svg.board(
        board=board,
        size=400,
        arrows=arrows,
        flipped=flipped,  # Brett drehen, wenn nötig
        coordinates=True
    )

    # Anzeige des SVG-Bildes in Streamlit
    st.components.v1.html(board_svg, height=500)

def rotate_board_and_midpoints(warped_image, midpoints, white_side):
    height, width = warped_image.shape[:2]
    if white_side == "Rechts":
        # Weiß spielt rechts, Brett um 90 Grad drehen
        rotated_image = cv2.rotate(warped_image, cv2.ROTATE_90_CLOCKWISE)
        rotated_midpoints = np.zeros_like(midpoints)
        rotated_midpoints[:, 0] = midpoints[:, 1]
        rotated_midpoints[:, 1] = width - midpoints[:, 0]
    elif white_side == "Links":
        # Weiß spielt links, Brett um 270 Grad drehen
        rotated_image = cv2.rotate(warped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_midpoints = np.zeros_like(midpoints)
        rotated_midpoints[:, 0] = height - midpoints[:, 1]
        rotated_midpoints[:, 1] = midpoints[:, 0]
    else:
        rotated_image = warped_image
        rotated_midpoints = midpoints
    return rotated_image, rotated_midpoints

def rotate_midpoints_180(midpoints, width, height):
    rotated_midpoints = np.zeros_like(midpoints)
    rotated_midpoints[:, 0] = width - midpoints[:, 0]
    rotated_midpoints[:, 1] = height - midpoints[:, 1]
    return rotated_midpoints

def save_pgn(game):
    # Speichern des Spiels als PGN-Datei
    with open('gespeichertes_spiel.pgn', 'w') as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
    st.write("Spiel wurde als 'gespeichertes_spiel.pgn' gespeichert.")

def main():
    st.title("Schachbrett- und Figuren-Erkennung aus Video")

    # Video hochladen
    uploaded_video = st.file_uploader("Lade ein Video des Schachbretts hoch", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Speichere das Video in einer temporären Datei
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        # Video lesen
        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("Fehler beim Öffnen des Videos.")
            return

        # Initialisierung
        previous_fen = None
        initial_position_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
        game_started = False  # Um zu verfolgen, ob wir von der Grundstellung ausgegangen sind

        # Benutzer wählt, ob Weiß links oder rechts spielt
        st.subheader("Spielerpositionen")
        white_side = st.selectbox("Auf welcher Seite spielt Weiß?", ("Links", "Rechts"))

        # Fortschrittsanzeige
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # Wir nehmen an, dass wir ein Frame pro Sekunde verarbeiten wollen
        progress_bar = st.progress(0)

        frame_number = 0
        game_over = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Video zu Ende

            frame_number += 1

            # Nur jedes n-te Frame verarbeiten
            if frame_number % frame_interval != 0:
                continue

            # Fortschrittsanzeige aktualisieren
            progress = frame_number / total_frames
            progress_bar.progress(min(progress, 1.0))

            # Bild verarbeiten
            image = frame
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Schritt 1: Figuren detektieren
            piece_midpoints, piece_labels, piece_results = detect_pieces(image)

            # Schritt 2: Schachbrett erkennen inkl. Korrekturvector
            detected_points, corner_results = detect_corners(image)
            A = detected_points["A"]
            B = detected_points["B"]
            C = detected_points["C"]
            D_calculated = calculate_point_D(A, B, C)
            D_corrected = adjust_point_D(A, B, C, D_calculated, PERCENT_AB, PERCENT_BC)
            sorted_points = sort_points(A, B, C, D_corrected.astype(int))
            warped_image, M = warp_perspective(image_rgb, sorted_points)

            # Transformiere die Mittelpunkte der Figuren
            ones = np.ones((piece_midpoints.shape[0], 1))
            piece_midpoints_homogeneous = np.hstack([piece_midpoints, ones])
            transformed_midpoints = M @ piece_midpoints_homogeneous.T
            transformed_midpoints /= transformed_midpoints[2, :]  # Homogenisierung
            transformed_midpoints = transformed_midpoints[:2, :].T  # Zurück zu kartesischen Koordinaten

            # Schritt 3: Brett drehen basierend auf Spielerposition
            rotated_warped_image, rotated_midpoints = rotate_board_and_midpoints(
                warped_image, transformed_midpoints, white_side
            )

            # Generiere die FEN-Notation
            current_fen_full = generate_fen_from_board(rotated_midpoints, piece_labels, player_to_move='w')
            current_fen = current_fen_full.split()[0]

            # Vergleichen, ob A1 unten links ist mit FEN der Grundstellung
            if current_fen == initial_position_fen:
                game_started = True
            else:
                # Brett um 180 Grad drehen und erneut prüfen
                rotated_warped_image = cv2.rotate(rotated_warped_image, cv2.ROTATE_180)
                rotated_midpoints = rotate_midpoints_180(rotated_midpoints, 800, 800)
                current_fen_full = generate_fen_from_board(rotated_midpoints, piece_labels, player_to_move='w')
                current_fen = current_fen_full.split()[0]
                if current_fen == initial_position_fen:
                    game_started = True
                else:
                    # Benutzerinput erforderlich
                    st.warning("Brett konnte nicht automatisch ausgerichtet werden.")
                    white_side = st.selectbox("Bitte geben Sie an, auf welcher Seite Weiß spielt.", ("Links", "Rechts"))
                    # Rotationslogik erneut anwenden
                    rotated_warped_image, rotated_midpoints = rotate_board_and_midpoints(
                        warped_image, transformed_midpoints, white_side
                    )
                    current_fen_full = generate_fen_from_board(rotated_midpoints, piece_labels, player_to_move='w')
                    current_fen = current_fen_full.split()[0]

            # Schritt 4: Uhr analysieren
            player_turn, clock_result = detect_player_turn(image)
            if player_turn == 'hold' or player_turn is None:
                # Uhr auf hold, springe zu nächstem Frame
                continue

            # Mappe 'left' und 'right' zu Spielern basierend auf der Position von Weiß
            if player_turn == 'left':
                if white_side == "Links":
                    player_to_move = 'w'
                else:
                    player_to_move = 'b'
            elif player_turn == 'right':
                if white_side == "Rechts":
                    player_to_move = 'w'
                else:
                    player_to_move = 'b'
            else:
                # Benutzerinput erforderlich
                player_input = st.selectbox("Bitte wählen Sie, wer am Zug ist:", ("Weiß", "Schwarz"))
                if player_input == "Weiß":
                    player_to_move = 'w'
                else:
                    player_to_move = 'b'

            # Brett und Figuren analysieren und vergleichen
            if current_fen == previous_fen:
                # Keine Änderung, springe zu nächstem Frame
                continue
            else:
                # Änderung festgestellt
                previous_fen = current_fen

                # Schritt 4b: Überprüfen auf Schachmatt
                fen_string = generate_fen_from_board(rotated_midpoints, piece_labels, player_to_move=player_to_move)
                board = chess.Board(fen_string)
                if board.is_checkmate():
                    # Schachmatt erkannt
                    game_over = True
                    st.success("Schachmatt erkannt!")
                    if game_started:
                        # Von Grundstellung ausgegangen, konvertiere FEN zu PGN
                        game = chess.pgn.Game.from_board(board)
                        st.write("Partie als PGN:")
                        st.text(str(game))
                    else:
                        st.warning("Partie begann nicht von der Grundstellung.")
                    break  # Beende die Schleife
                else:
                    # Nicht Schachmatt, fahre fort
                    pass

        cap.release()
        progress_bar.empty()

        if not game_over:
            st.write("Videoverarbeitung abgeschlossen, kein Schachmatt erkannt.")
        else:
            # Schritt 5: Glückwunschnachricht
            st.write("Glückwunsch! Das Spiel ist beendet.")

            # Schritt 6: Spiel sichern
            save_pgn(game)

    else:
        st.write("Bitte laden Sie ein Video des Schachbretts hoch.")

if __name__ == "__main__":
    main()
