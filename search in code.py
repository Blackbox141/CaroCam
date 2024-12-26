import numpy as np
import cv2
from ultralytics import YOLO
import os
import requests
import tempfile
import io
import webbrowser
import time
import datetime
import matplotlib.pyplot as plt
import logging

# python-chess (für die echte PGN-Generierung)
import chess
import chess.pgn

# Unterdrücke ggf. die ultralytics-Logs:
logging.getLogger("ultralytics").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

piece_model_path = os.path.join(
    BASE_DIR, 'Figure Detection 3', 'runs', 'detect', 'yolov8-chess32', 'weights', 'best.pt'
)
corner_model_path = os.path.join(
    BASE_DIR, 'Corner Detection 2', 'runs', 'detect', 'yolov8n_corner14', 'weights', 'best.pt'
)
clock_model_path = os.path.join(
    BASE_DIR, 'Clock Detection 3', 'runs', 'detect', 'yolov8-chess6', 'weights', 'best.pt'
)

if os.path.exists(piece_model_path) and os.path.exists(corner_model_path) and os.path.exists(clock_model_path):
    piece_model = YOLO(piece_model_path)
    corner_model = YOLO(corner_model_path)
    clock_model = YOLO(clock_model_path)
else:
    print("Modelldatei nicht gefunden.")
    exit()

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

PERCENT_AB = 0.17
PERCENT_BC = -0.07

STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

def detect_pieces(image, min_conf=0.7):
    """
    Erkennen aller Schachfiguren im Bild mittels YOLO-Modell.
    Nur Figuren akzeptieren, deren Confidence >= min_conf.
    Gibt midpoints, labels und confidences zurück.
    """
    results = piece_model.predict(
        image,
        conf=0.05,  # bewusst niedriger, damit YOLO viele Boxen anbietet
        iou=0.3,
        imgsz=1400,
        verbose=False
    )
    class_names = piece_model.model.names
    result = results[0]

    midpoints = []
    labels = []
    confidences = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls.cpu().numpy()[0])
        conf_val = float(box.conf.cpu().numpy()[0])
        label = class_names[cls_id]

        # Erkennungen unter min_conf ignorieren
        if conf_val < min_conf:
            continue

        mid_x = x1 + (x2 - x1) / 2
        mid_y = y1 + (y2 - y1) * 0.75

        midpoints.append([mid_x, mid_y])
        labels.append(label)
        confidences.append(conf_val)

    return np.array(midpoints), labels, confidences

def detect_corners(image):
    """
    Erkennen der 3 Eckpunkte (A, B, C) des Schachbretts.
    """
    results = corner_model(
        image,
        conf=0.1,
        iou=0.3,
        imgsz=1400,
        verbose=False
    )
    points = {}
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        cls_id = int(box.cls.cpu().numpy()[0])
        if cls_id == 0:
            points["A"] = np.array([center_x, center_y])
        elif cls_id == 1:
            points["B"] = np.array([center_x, center_y])
        elif cls_id == 2:
            points["C"] = np.array([center_x, center_y])

    if len(points) != 3:
        return None
    return points

def detect_player_turn(image, visualize=False):
    """
    Gibt IMMER ZWEI Werte zurück:
      (player_turn, results)  oder (player_turn, None)
    """
    results = clock_model(
        image,
        conf=0.1,
        iou=0.3,
        imgsz=1400,
        verbose=False
    )
    class_names = clock_model.model.names
    boxes = results[0].boxes

    labels = []
    for box in boxes:
        cls_id = int(box.cls.cpu().numpy()[0])
        label = class_names[cls_id]
        labels.append(label)

    if 'left' in labels:
        player_turn = 'left'
    elif 'right' in labels:
        player_turn = 'right'
    else:
        player_turn = None

    if 'hold' in labels:
        player_turn = 'hold'

    if visualize:
        return player_turn, results
    else:
        return player_turn, None

def plot_clock(image, results, relevant_label):
    """
    Optionale Visualisierung der Schachuhr (z.B. 'left' oder 'right').
    """
    display_img = image.copy()
    class_names = clock_model.model.names

    for box in results[0].boxes:
        cls_id = int(box.cls.cpu().numpy()[0])
        label = class_names[cls_id]
        conf = float(box.conf.cpu().numpy()[0])

        if label == relevant_label:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            text = f"{label} {conf * 100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 3
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = int(x1)
            text_y = int(y1) - 10

            cv2.rectangle(
                display_img,
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 255), -1
            )
            cv2.putText(
                display_img, text, (text_x, text_y),
                font, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA
            )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Erkannte Schachuhr", fontsize=16)
    ax.axis('off')
    plt.show()

def calculate_point_D(A, B, C):
    BC = C - B
    D_calculated = A + BC
    return D_calculated

def adjust_point_D(A, B, C, D_calculated, percent_AB, percent_BC):
    AB = B - A
    BC = C - B
    correction_vector = percent_AB * AB + percent_BC * BC
    D_corrected = D_calculated + correction_vector
    return D_corrected

def sort_points(A, B, C, D):
    points = np.array([A, B, C, D])
    points = sorted(points, key=lambda x: x[1])
    top_points = sorted(points[:2], key=lambda x: x[0])
    bottom_points = sorted(points[2:], key=lambda x: x[0])
    A_sorted, D_sorted = top_points
    B_sorted, C_sorted = bottom_points
    return np.array([A_sorted, B_sorted, C_sorted, D_sorted], dtype=np.float32)

def warp_perspective(image, src_points):
    """
    Entzerrt das Bild auf 800x800 und dreht es um 180°,
    damit die visuelle Anzeige nicht auf dem Kopf steht.
    """
    dst_size = 800
    dst_points = np.array([
        [0, 0],
        [0, dst_size - 1],
        [dst_size - 1, dst_size - 1],
        [dst_size - 1, 0]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, M, (dst_size, dst_size))

    # 180°-Drehung
    warped_image = cv2.rotate(warped_image, cv2.ROTATE_180)

    return warped_image, M

def plot_final_board(image, midpoints, labels, confidences):
    """
    Zeichnet ein 8x8-Raster nur dann, wenn mind. 1 Figur erkannt wurde.
    Zeigt zusätzlich die Confidence an, z.B. "Q(85.3%)".
    """
    if midpoints.shape[0] == 0:
        return None

    grid_size = 8
    step_size = image.shape[0] // grid_size
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image, extent=[0, image.shape[1], image.shape[0], 0])

    for i in range(grid_size + 1):
        ax.axhline(i * step_size, color='black', linewidth=1)
        ax.axvline(i * step_size, color='black', linewidth=1)

    for i in range(grid_size):
        ax.text(i * step_size + step_size / 2,
                image.shape[0] - 10,
                chr(65 + i),
                fontsize=12, color='black',
                ha='center', va='center')
        ax.text(10,
                i * step_size + step_size / 2,
                str(grid_size - i),
                fontsize=12, color='black',
                ha='center', va='center')

    for (point, label, conf_val) in zip(midpoints, labels, confidences):
        x, y = point
        col = int(x // step_size)
        row = int(y // step_size)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            fen_char = FEN_MAPPING.get(label, '?')
            square_x = col * step_size + step_size / 2
            square_y = row * step_size + step_size / 2
            text_str = f"{fen_char}({conf_val*100:.1f}%)"
            ax.text(square_x, square_y, text_str, fontsize=16, color='red', ha='center', va='center')

    ax.set_title("Digitales Schachbrett mit erkannten Figuren und Confidence", fontsize=16)
    return fig

def generate_placement_from_board(midpoints, labels, grid_size=8):
    board = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    step_size = 800 // grid_size

    for point, label in zip(midpoints, labels):
        x, y = point
        col = int(y // step_size)
        row = 7 - int(x // step_size)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            fen_char = FEN_MAPPING.get(label, '')
            board[row][col] = fen_char

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

    fen_string = '/'.join(fen_rows)
    return fen_string


def fen_diff_to_move(fen1, fen2, color='w'):
    """
    Try-all-legal-moves:
    Baue aus fen1 + color + KQkq - 0 1 ein Schachbrett,
    teste jeden legalen Zug, ob er fen2 ergibt.
    """
    if not fen1 or not fen2:
        return None

    # Rochaderechte "KQkq"
    full_fen1 = f"{fen1} {color} KQkq - 0 1"

    try:
        board = chess.Board(full_fen1)
    except:
        return None

    for move in board.legal_moves:
        board.push(move)
        fen_part = board.fen().split(" ")[0]
        if fen_part == fen2:
            return move
        board.pop()

    return None

def convert_fen_list_to_pgn(fen_history):
    """
    Erzeugt aus einer Folge von FENs eine PGN, indem je zwei FENs
    via 'fen_diff_to_move' in einen legalen Zug übersetzt werden.
    Kein Null-Zug + kein Kommentar -> "saubere" PGN.

    NEU: Quick-Fix:
    - Wir pflegen ein global_board, um die gesamte Partie fortlaufend
      legal zu halten.
    - Falls der Zug laut "global_board.legal_moves" nicht legal ist,
      wird er geskippt und FEN2 ignoriert.
    """
    game = chess.pgn.Game()
    node = game

    # global_board repräsentiert unsere gesamte fortlaufende Partie
    global_board = chess.Board()  # Standardstart
    color = 'w'  # wir toggeln w->b->w->b

    for i in range(len(fen_history) - 1):
        fen1 = fen_history[i]["fen"]
        fen2 = fen_history[i + 1]["fen"]

        move = fen_diff_to_move(fen1, fen2, color=color)
        if move:
            # Test: Ist move im globalen Partieverlauf legal?
            if move in global_board.legal_moves:
                global_board.push(move)
                node = node.add_variation(move)

                # Farbe nur toggeln, wenn ein Move ausgeführt wurde
                color = 'b' if color == 'w' else 'w'
            else:
                print(f"Skipping move {move}, nicht legal in global progression. Ignoriere fen2: {fen2}")
                # => KEIN color toggle, KEIN push
                # => Nächster Schleifendurchlauf vergleicht fen[i+1] mit fen[i+2]
        else:
            print(f"Kein Move gefunden für fen {fen1}->{fen2}, color={color}, skippe fen2.")
            # => Kein push, kein color toggle

    return game


def main():
    print("Schachbrett- und Figuren-Erkennung aus Video (FEN->PGN via Try-all-legal-moves, 180°-Rotation, Quick-Fix)")

    video_path = input("Gib den Pfad zum Schachspiel-Video ein: ")
    if not os.path.exists(video_path):
        print("Videodatei nicht gefunden.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Fehler beim Laden des Videos.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    frame_interval = max(1, int(fps * 0.2))

    print("Versuche, Eckpunkte zu erkennen...")
    corners_detected = False
    while not corners_detected:
        ret, frame = cap.read()
        if not ret:
            print("Keine weiteren Frames verfügbar. Eckpunkte konnten nicht erkannt werden.")
            return

        detected_points = detect_corners(frame)
        if detected_points is not None:
            corners_detected = True
            A = detected_points["A"]
            B = detected_points["B"]
            C = detected_points["C"]
            print("Eckpunkte erkannt.")

    # Vierter Eckpunkt
    D_calculated = calculate_point_D(A, B, C)
    D_corrected = adjust_point_D(A, B, C, D_calculated, PERCENT_AB, PERCENT_BC)
    sorted_pts = sort_points(A, B, C, D_corrected.astype(int))

    # Matrix für Perspektiv-Transformation
    M = cv2.getPerspectiveTransform(
        sorted_pts,
        np.array([[0, 0], [0, 799], [799, 799], [799, 0]], dtype=np.float32)
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    previous_player_turn = None
    previous_placement = None
    fen_history = []

    white_side = None
    user_white_side = None
    game_over = False

    while cap.isOpened() and not game_over:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % frame_interval != 0:
            continue

        player_turn, clock_results = detect_player_turn(frame, visualize=False)

        if player_turn is not None and player_turn != 'hold' and player_turn != previous_player_turn:
            print(f"Uhrposition geändert: {previous_player_turn} -> {player_turn}")

            midpoints, labels, confidences = detect_pieces(frame, min_conf=0.7)
            if midpoints.shape[0] > 0:
                ones = np.ones((midpoints.shape[0], 1))
                midpoints_hom = np.hstack([midpoints, ones])
                transformed_midpoints = M @ midpoints_hom.T
                transformed_midpoints /= transformed_midpoints[2, :]
                transformed_midpoints = transformed_midpoints[:2, :].T

                if white_side is None:
                    fen_found = False
                    for side in ["Links", "Rechts"]:
                        if side == "Links":
                            rotated_midpoints = transformed_midpoints.copy()
                            rotated_labels = labels[:]
                            rotated_confidences = confidences[:]
                        else:
                            rotated_midpoints = np.zeros_like(transformed_midpoints)
                            rotated_midpoints[:, 0] = 800 - transformed_midpoints[:, 0]
                            rotated_midpoints[:, 1] = 800 - transformed_midpoints[:, 1]
                            rotated_labels = labels[:]
                            rotated_confidences = confidences[:]

                        current_placement = generate_placement_from_board(rotated_midpoints, rotated_labels)
                        if current_placement == STARTING_PLACEMENT:
                            white_side = side
                            fen_found = True
                            previous_placement = current_placement
                            print(f"Weiß erkannt auf {white_side}")
                            break

                    if not fen_found:
                        if user_white_side is None:
                            print("Konnte nicht feststellen, wo Weiß ist.")
                            user_white_side = input("Bitte wählen: Links/Rechts? ")
                            if user_white_side not in ["Links", "Rechts"]:
                                user_white_side = "Links"
                            white_side = user_white_side

                        if white_side == "Links":
                            rotated_midpoints = transformed_midpoints.copy()
                            rotated_labels = labels[:]
                            rotated_confidences = confidences[:]
                        else:
                            rotated_midpoints = np.zeros_like(transformed_midpoints)
                            rotated_midpoints[:, 0] = 800 - transformed_midpoints[:, 0]
                            rotated_midpoints[:, 1] = 800 - transformed_midpoints[:, 1]
                            rotated_labels = labels[:]
                            rotated_confidences = confidences[:]

                        current_placement = generate_placement_from_board(rotated_midpoints, rotated_labels)
                        if previous_placement is None:
                            previous_placement = current_placement
                            fen_history.append({"fen": current_placement, "changed": False})
                else:
                    if white_side == "Links":
                        rotated_midpoints = transformed_midpoints.copy()
                        rotated_labels = labels[:]
                        rotated_confidences = confidences[:]
                    else:
                        rotated_midpoints = np.zeros_like(transformed_midpoints)
                        rotated_midpoints[:, 0] = 800 - transformed_midpoints[:, 0]
                        rotated_midpoints[:, 1] = 800 - transformed_midpoints[:, 1]
                        rotated_labels = labels[:]
                        rotated_confidences = confidences[:]

                    current_placement = generate_placement_from_board(rotated_midpoints, rotated_labels)

                    if previous_placement is not None and current_placement != previous_placement:
                        print("FEN hat sich geändert.")
                        warped_image, _ = warp_perspective(frame, sorted_pts)
                        fig_board = plot_final_board(
                            warped_image,
                            rotated_midpoints,
                            rotated_labels,
                            rotated_confidences
                        )
                        if fig_board:
                            plt.show()

                        fen_history.append({"fen": current_placement, "changed": True})
                        previous_placement = current_placement
                    else:
                        fen_history.append({"fen": current_placement, "changed": False})
                        print("FEN hat sich NICHT geändert.")
                        if previous_placement is None:
                            previous_placement = current_placement

        previous_player_turn = player_turn

    cap.release()
    print("Videoverarbeitung abgeschlossen.")

    print("\n=== Auflistung aller erstellten FENs ===")
    for i, fen_item in enumerate(fen_history, start=1):
        changed_str = "JA" if fen_item["changed"] else "NEIN"
        print(f"{i}. {fen_item['fen']} (Änderung: {changed_str})")

    # PGN aus FEN-Liste, mit Quick-Fix (illegale Züge -> skippen)
    game = convert_fen_list_to_pgn(fen_history)
    print("\n=== PGN (ohne Kommentare oder Null-Züge, illegal = skip) ===")
    print(game)

if __name__ == "__main__":
    main()
