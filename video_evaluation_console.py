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
    BASE_DIR, 'Figure Detection Model', 'runs', 'detect', 'yolov8-chess32', 'weights', 'best.pt'
)
corner_model_path = os.path.join(
    BASE_DIR, 'Corner Detection Model', 'runs', 'detect', 'yolov8n_corner14', 'weights', 'best.pt'
)
clock_model_path = os.path.join(
    BASE_DIR, 'Clock Detection Model', 'runs', 'detect', 'yolov8-chess6', 'weights', 'best.pt'
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

        if conf_val < min_conf:
            continue

        mid_x = x1 + (x2 - x1) / 2
        mid_y = y1 + (y2 - y1) * 0.75  # kleiner offset nach unten

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
    warped_image = cv2.rotate(warped_image, cv2.ROTATE_180)
    return warped_image, M


def plot_final_board(image, midpoints, labels, confidences):
    """
    Zeichnet ein 8x8-Raster auf das gegebene (entzerrte) Bild
    und schreibt an die erkannten Positionen die Figur + Confidence in %.
    """
    if midpoints.shape[0] == 0:
        return None

    grid_size = 8
    step_size = image.shape[0] // grid_size

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, extent=[0, image.shape[1], image.shape[0], 0])

    for i in range(grid_size + 1):
        ax.axhline(i * step_size, color='black', linewidth=1)
        ax.axvline(i * step_size, color='black', linewidth=1)

    # Spalten-Buchstaben
    for i in range(grid_size):
        ax.text(i * step_size + step_size / 2,
                image.shape[0] - 10,
                chr(65 + i),
                fontsize=12, color='black',
                ha='center', va='center')
        # Reihen-Zahlen
        ax.text(10,
                i * step_size + step_size / 2,
                str(grid_size - i),
                fontsize=12, color='black',
                ha='center', va='center')

    for (x, y), lbl, conf_val in zip(midpoints, labels, confidences):
        col = int(x // step_size)
        row = int(y // step_size)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            fen_char = FEN_MAPPING.get(lbl, '?')
            square_x = col * step_size + step_size / 2
            square_y = row * step_size + step_size / 2
            text_str = f"{fen_char}({conf_val * 100:.1f}%)"
            ax.text(square_x, square_y,
                    text_str,
                    fontsize=14, color='red',
                    ha='center', va='center')

    ax.set_title("Erkannte Figuren (Confidence) auf dem entzerrten Brett", fontsize=12)
    ax.axis('off')
    return fig


def generate_placement_from_board(midpoints, labels, grid_size=8):
    board = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    step_size = 800 // grid_size

    for (x, y), lbl in zip(midpoints, labels):
        col = int(y // step_size)
        row = 7 - int(x // step_size)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            fen_char = FEN_MAPPING.get(lbl, '')
            board[row][col] = fen_char

    fen_rows = []
    for row_data in board:
        empty_count = 0
        fen_row = ''
        for square in row_data:
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

    return '/'.join(fen_rows)


def fen_diff_to_move(fen1, fen2, color='w'):
    """
    Try-all-legal-moves:
    Baue aus fen1 + color + KQkq - 0 1 ein Schachbrett,
    teste jeden legalen Zug, ob er fen2 ergibt.
    """
    if not fen1 or not fen2:
        return None

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


def fix_fen_with_single_frames_on_the_fly(cap, current_frame_index, frame_interval,
                                          M, white_side,
                                          old_fen, color,
                                          max_tries=10,
                                          min_conf=0.7):
    """
    Versucht, durch NACHEINANDERLESEN EINZELNER FRAMES
    eine neue FEN zu erhalten, die im Vergleich zu old_fen
    einen legalen Zug bildet.

    Ablauf:
      - bis zu max_tries mal den nächsten Frame laden
      - Figurenerkennung -> FEN
      - wenn fen_diff_to_move(old_fen, new_fen) legal, return (True, new_fen)
      - sonst weiter
    Falls kein Erfolg: return (False, old_fen)
    """

    start_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    used_frames = 0
    success_fen = None

    for _ in range(max_tries):
        next_frame_pos = current_frame_index + frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_pos)
        current_frame_index = next_frame_pos

        ret, frame = cap.read()
        if not ret:
            # Keine weiteren Frames
            break

        # Erkennung
        midpoints, labels, confidences = detect_pieces(frame, min_conf=min_conf)
        if midpoints.shape[0] == 0:
            used_frames += 1
            continue

        # Perspektivische Transformation
        ones = np.ones((midpoints.shape[0], 1))
        midpoints_hom = np.hstack([midpoints, ones])
        transformed = M @ midpoints_hom.T
        transformed /= transformed[2, :]
        transformed = transformed[:2, :].T

        # Falls Weiss rechts, drehen
        if white_side == "Rechts":
            rotated_mid = np.zeros_like(transformed)
            rotated_mid[:, 0] = 800 - transformed[:, 0]
            rotated_mid[:, 1] = 800 - transformed[:, 1]
            rotated_labels = labels[:]
            rotated_conf = confidences[:]
        else:
            rotated_mid = transformed
            rotated_labels = labels[:]
            rotated_conf = confidences[:]

        new_fen = generate_placement_from_board(rotated_mid, rotated_labels)

        move = fen_diff_to_move(old_fen, new_fen, color=color)
        if move:
            # => legaler Zug
            success_fen = new_fen
            break

        used_frames += 1

    # Wieder Framepointer zurück
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

    if success_fen is not None:
        return True, success_fen
    else:
        return False, old_fen


def main():
    print("Video -> Schachzug-Erkennung mit Frame-für-Frame-Fix (kein Mehrheitsabgleich)")

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

    # --- Eckpunkte-Erkennung (A, B, C) ---
    corners_detected = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detected_points = detect_corners(frame)
        if detected_points is not None:
            corners_detected = True
            A = detected_points["A"]
            B = detected_points["B"]
            C = detected_points["C"]
            print("Eckpunkte erkannt.")
            break

    if not corners_detected:
        print("Keine Ecken erkannt. Abbruch.")
        return

    D_calc = calculate_point_D(A, B, C)
    D_corr = adjust_point_D(A, B, C, D_calc, PERCENT_AB, PERCENT_BC)
    sorted_pts = sort_points(A, B, C, D_corr.astype(int))

    M = cv2.getPerspectiveTransform(
        sorted_pts,
        np.array([[0, 0], [0, 799], [799, 799], [799, 0]], dtype=np.float32)
    )

    # Zurückspulen (wir wollen ab Frame 0 erneut loslegen)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    previous_player_turn = None
    previous_fen = None
    color = 'w'  # abwechselnd w/b

    white_side = None
    user_white_side = None

    # PGN
    game = chess.pgn.Game()
    node = game
    global_board = chess.Board()  # Standard-Start

    while True:
        frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break  # Video zu Ende

        current_frame_index = int(frame_pos)

        # Nur alle frame_interval Frames
        if current_frame_index % frame_interval != 0:
            continue

        # Uhr abfragen
        player_turn, _ = detect_player_turn(frame, visualize=False)

        # Wenn sich die Uhrposition ändert und NICHT 'hold'
        if player_turn is not None and player_turn != 'hold' and player_turn != previous_player_turn:
            print(f"[Frame {current_frame_index}] Uhr geändert: {previous_player_turn} -> {player_turn}")

            midpoints, labels, confidences = detect_pieces(frame, min_conf=0.7)
            if midpoints.shape[0] > 0:
                # Perspektive
                ones = np.ones((midpoints.shape[0], 1))
                midpoints_hom = np.hstack([midpoints, ones])
                transformed = M @ midpoints_hom.T
                transformed /= transformed[2, :]
                transformed = transformed[:2, :].T

                if white_side is None:
                    fen_found = False
                    for side in ["Links", "Rechts"]:
                        if side == "Links":
                            rotated_mid = transformed.copy()
                            rotated_labels = labels[:]
                            rotated_conf = confidences[:]
                        else:
                            rotated_mid = np.zeros_like(transformed)
                            rotated_mid[:, 0] = 800 - transformed[:, 0]
                            rotated_mid[:, 1] = 800 - transformed[:, 1]
                            rotated_labels = labels[:]
                            rotated_conf = confidences[:]

                        test_fen = generate_placement_from_board(rotated_mid, rotated_labels)
                        if test_fen == STARTING_PLACEMENT:
                            white_side = side
                            fen_found = True
                            previous_fen = test_fen
                            global_board = chess.Board()
                            print(f"Weiss erkannt auf {white_side}, Startstellung bestätigt.")
                            # Optional Visualisierung
                            warped_image, _ = warp_perspective(frame, sorted_pts)
                            fig = plot_final_board(warped_image, rotated_mid, rotated_labels, rotated_conf)
                            if fig:
                                plt.show()
                            break

                    if not fen_found:
                        # Nutzer fragen
                        if user_white_side is None:
                            print("Konnte nicht automatisch feststellen, wo Weiss ist.")
                            user_white_side = input("Bitte wählen: Links/Rechts? ")
                            if user_white_side not in ["Links", "Rechts"]:
                                user_white_side = "Links"
                            white_side = user_white_side

                        if white_side == "Rechts":
                            rotated_mid = np.zeros_like(transformed)
                            rotated_mid[:, 0] = 800 - transformed[:, 0]
                            rotated_mid[:, 1] = 800 - transformed[:, 1]
                            rotated_labels = labels[:]
                            rotated_conf = confidences[:]
                        else:
                            rotated_mid = transformed.copy()
                            rotated_labels = labels[:]
                            rotated_conf = confidences[:]

                        test_fen = generate_placement_from_board(rotated_mid, rotated_labels)
                        if previous_fen is None:
                            previous_fen = test_fen
                            # Optional Visualisierung
                            warped_image, _ = warp_perspective(frame, sorted_pts)
                            fig = plot_final_board(warped_image, rotated_mid, rotated_labels, rotated_conf)
                            if fig:
                                plt.show()

                else:
                    # Weiss-Seite bekannt
                    if white_side == "Rechts":
                        rotated_mid = np.zeros_like(transformed)
                        rotated_mid[:, 0] = 800 - transformed[:, 0]
                        rotated_mid[:, 1] = 800 - transformed[:, 1]
                        rotated_labels = labels[:]
                        rotated_conf = confidences[:]
                    else:
                        rotated_mid = transformed.copy()
                        rotated_labels = labels[:]
                        rotated_conf = confidences[:]

                    current_fen = generate_placement_from_board(rotated_mid, rotated_labels)
                    if previous_fen is not None and current_fen != previous_fen:
                        print(f"Neue FEN: {current_fen}")
                        # Visualisierung
                        warped_image, _ = warp_perspective(frame, sorted_pts)
                        fig = plot_final_board(warped_image, rotated_mid, rotated_labels, rotated_conf)
                        if fig:
                            plt.show()

                        move = fen_diff_to_move(previous_fen, current_fen, color=color)
                        if move and move in global_board.legal_moves:
                            global_board.push(move)
                            node = node.add_variation(move)
                            print(f"Legal: {move} (color={color}) -> Board nun {global_board.fen()}")
                            color = 'b' if color == 'w' else 'w'
                            previous_fen = current_fen
                        else:
                            # Kein legaler Zug => Frame-für-Frame-Fix
                            print("Kein legaler Zug -> Versuche Fix mit nächsten Einzel-Frames ...")
                            fixed_ok, fixed_fen = fix_fen_with_single_frames_on_the_fly(
                                cap, current_frame_index, frame_interval,
                                M, white_side,
                                old_fen=previous_fen, color=color,
                                max_tries=10, min_conf=0.7
                            )
                            if fixed_ok:
                                move2 = fen_diff_to_move(previous_fen, fixed_fen, color=color)
                                if move2 and move2 in global_board.legal_moves:
                                    global_board.push(move2)
                                    node = node.add_variation(move2)
                                    print(f"Einzel-Frame-Korrektur erfolgreich: {move2} (color={color})")
                                    color = 'b' if color == 'w' else 'w'
                                    previous_fen = fixed_fen
                                else:
                                    print("Trotz Korrektur kein legaler Zug. FEN ignoriert.")
                            else:
                                print("Frame-für-Frame-Korrektur konnte den Zug nicht reparieren. FEN ignoriert.")
                    else:
                        # Falls previous_fen None ist oder FEN sich nicht ändert
                        if previous_fen is None:
                            previous_fen = current_fen

        previous_player_turn = player_turn

    cap.release()

    print("\n=== Resultierendes PGN ===")
    print(game)
    print("Fertig.")


if __name__ == "__main__":
    main()
