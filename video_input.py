import numpy as np
import cv2
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import chess
import chess.pgn
import logging

# Logging unterdrücken für ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# An deine Pfade anpassen
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

# Mapping von YOLO-Klassen zu FEN-Buchstaben
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

# Standard-Aufstellung (FEN)
STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

# Zwei Korrektur-Parameter für den vierten Eckpunkt
PERCENT_AB = 0.17
PERCENT_BC = -0.07


def detect_pieces(image, min_conf=0.7):
    """Erkennt Schachfiguren per YOLO."""
    results = piece_model.predict(
        image,
        conf=0.05,  # absichtlich niedrig, damit YOLO mehr Boxen anbietet
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

        # Mittelpunkt in x
        mid_x = x1 + (x2 - x1) / 2
        # Mittelpunkt in y, aber etwas tiefer:
        mid_y = y1 + (y2 - y1) * 0.75

        midpoints.append([mid_x, mid_y])
        labels.append(label)
        confidences.append(conf_val)

    return np.array(midpoints), labels, confidences


def detect_corners(image):
    """Erkennt die 3 Eckpunkte A, B, C per YOLO."""
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
    """Erkennt mit clock_model: 'left', 'right', 'hold' oder None."""
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

    player_turn = None
    if 'left' in labels:
        player_turn = 'left'
    elif 'right' in labels:
        player_turn = 'right'
    if 'hold' in labels:
        player_turn = 'hold'

    if visualize:
        return player_turn, results
    else:
        return player_turn, None


def calculate_point_D(A, B, C):
    """Rechnet den vierten Eckpunkt D = A + (C - B)."""
    BC = C - B
    return A + BC


def adjust_point_D(A, B, C, D_calculated, percent_AB, percent_BC):
    """Korrigiert D um eine kleine Verschiebung basierend auf Vektoren AB und BC."""
    AB = B - A
    BC = C - B
    correction_vector = percent_AB * AB + percent_BC * BC
    return D_calculated + correction_vector


def sort_points(A, B, C, D):
    """Sortiert die vier Eckpunkte für OpenCVs getPerspectiveTransform."""
    pts = np.array([A, B, C, D])
    pts = sorted(pts, key=lambda x: x[1])
    top_pts = sorted(pts[:2], key=lambda x: x[0])
    bot_pts = sorted(pts[2:], key=lambda x: x[0])
    A_sorted, D_sorted = top_pts
    B_sorted, C_sorted = bot_pts
    return np.array([A_sorted, B_sorted, C_sorted, D_sorted], dtype=np.float32)


def warp_perspective(image, src_points):
    """Entzerrt das Bild auf 800x800 und dreht um 180°, damit es nicht auf dem Kopf steht."""
    dst_size = 800
    dst_points = np.array([
        [0, 0],
        [0, dst_size - 1],
        [dst_size - 1, dst_size - 1],
        [dst_size - 1, 0]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, (dst_size, dst_size))
    # 180°-Rotation
    warped = cv2.rotate(warped, cv2.ROTATE_180)
    return warped, M


def plot_final_board(image, midpoints, labels, confidences):
    """
    Zeichnet ein 8x8-Gitter auf das entzerrte Bild,
    schreibt an die erkannte Figur-Position die FEN-Figur + Confidence.
    """
    if midpoints.shape[0] == 0:
        return None

    grid_size = 8
    step_size = image.shape[0] // grid_size

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, extent=[0, image.shape[1], image.shape[0], 0])

    # Gitterlinien
    for i in range(grid_size + 1):
        ax.axhline(i * step_size, color='black', linewidth=1)
        ax.axvline(i * step_size, color='black', linewidth=1)

    # Spalten (A..H) und Reihen (1..8)
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

    for (x, y), lbl, conf_val in zip(midpoints, labels, confidences):
        col = int(x // step_size)
        row = int(y // step_size)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            fen_char = FEN_MAPPING.get(lbl, '?')
            cx = col * step_size + step_size / 2
            cy = row * step_size + step_size / 2
            text_str = f"{fen_char}({conf_val*100:.1f}%)"
            ax.text(cx, cy, text_str,
                    fontsize=14, color='red',
                    ha='center', va='center')

    ax.set_title("Erkannte Figuren (Confidence) auf dem entzerrten Brett", fontsize=12)
    ax.axis('off')
    return fig


def generate_placement_from_board(midpoints, labels, grid_size=8):
    """
    Wandelt (x,y)+Label in eine FEN-Zeichenkette um.
    (x→col, y→row) mit Umkehrung row = 7 - (x//step_size)
    """
    board = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    step_size = 800 // grid_size
    for (mx, my), lbl in zip(midpoints, labels):
        c = int(my // step_size)
        r = 7 - int(mx // step_size)
        if 0 <= r < grid_size and 0 <= c < grid_size:
            fen_char = FEN_MAPPING.get(lbl, '')
            board[r][c] = fen_char

    fen_rows = []
    for row_data in board:
        empty_count = 0
        fen_row = ''
        for sq in row_data:
            if sq == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += sq
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    return '/'.join(fen_rows)


def fen_diff_to_move(fen1, fen2, color='w'):
    """Prüft alle legalen Züge von fen1 aus (python-chess), ob fen2 erzeugt wird."""
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
    Versucht nacheinander in den nächsten max_tries Frames,
    eine FEN zu finden, die old_fen in einen legalen Zug überführt.
    Sobald eine legale FEN gefunden wird, return (True, new_fen).
    Ansonsten (False, old_fen).
    """
    start_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    success_fen = None
    for _ in range(max_tries):
        next_frame_pos = current_frame_index + frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_pos)
        current_frame_index = next_frame_pos

        ret, frame = cap.read()
        if not ret:
            break

        midpoints, labels, confidences = detect_pieces(frame, min_conf=min_conf)
        if midpoints.shape[0] == 0:
            continue

        # Perspektive transformieren
        ones = np.ones((midpoints.shape[0], 1))
        mid_hom = np.hstack([midpoints, ones])
        transformed = M @ mid_hom.T
        transformed /= transformed[2, :]
        transformed = transformed[:2, :].T

        if white_side == "Rechts":
            rotated = np.zeros_like(transformed)
            rotated[:, 0] = 800 - transformed[:, 0]
            rotated[:, 1] = 800 - transformed[:, 1]
            rotated_labels = labels[:]
        else:
            rotated = transformed
            rotated_labels = labels[:]

        new_fen = generate_placement_from_board(rotated, rotated_labels)
        move = fen_diff_to_move(old_fen, new_fen, color=color)
        if move:
            success_fen = new_fen
            break

    # Framepointer zurücksetzen, damit wir im Haupt-Loop nicht "vorspringen"
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

    if success_fen is not None:
        return True, success_fen
    else:
        return False, old_fen


def main():
    print("=== Video -> Schachzug-Erkennung (Frame-für-Frame Suche nach Grundstellung & Single-Frame-Fix) ===")

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

    # 0.1 bis 0.2 Sekunden (Beispiel)
    frame_interval = max(1, int(fps * 0.2))

    # 1) Ecken finden
    corners_found = False
    M = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cpoints = detect_corners(frame)
        if cpoints is not None:
            corners_found = True
            A = cpoints["A"]
            B = cpoints["B"]
            C = cpoints["C"]
            D_calc = calculate_point_D(A, B, C)
            D_corr = adjust_point_D(A, B, C, D_calc, PERCENT_AB, PERCENT_BC)
            sorted_pts = sort_points(A, B, C, D_corr.astype(int))
            M = cv2.getPerspectiveTransform(
                sorted_pts,
                np.array([[0, 0], [0, 799], [799, 799], [799, 0]], dtype=np.float32)
            )
            print("Eckpunkte erkannt, M berechnet.")
            break

    if not corners_found or M is None:
        print("Keine Ecken erkannt. Abbruch.")
        return

    # 2) Suche nach Standard-Aufstellung (Grundstellung) -> Wo spielt Weiß?
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    white_side = None
    found_start_position = False

    while True:
        ret, frame = cap.read()
        if not ret:
            # Video zu Ende -> nicht gefunden
            break

        midpoints, labels, _ = detect_pieces(frame, min_conf=0.7)
        if midpoints.shape[0] == 0:
            continue

        # Perspektive
        ones = np.ones((midpoints.shape[0], 1))
        mid_hom = np.hstack([midpoints, ones])
        transformed = M @ mid_hom.T
        transformed /= transformed[2, :]
        transformed = transformed[:2, :].T

        # Teste beide Seiten: "Links" vs. "Rechts"
        for side in ["Links", "Rechts"]:
            if side == "Rechts":
                test_mid = np.zeros_like(transformed)
                test_mid[:, 0] = 800 - transformed[:, 0]
                test_mid[:, 1] = 800 - transformed[:, 1]
            else:
                test_mid = transformed

            test_fen = generate_placement_from_board(test_mid, labels)
            if test_fen == STARTING_PLACEMENT:
                white_side = side
                print(f"Grundstellung erkannt: Weiß spielt auf {white_side}.")
                found_start_position = True
                break

        if found_start_position:
            break

    if not found_start_position:
        # User fragen, wenn man es nicht automatisch herausgefunden hat
        print("Konnte nicht automatisch feststellen, wo Weiß spielt. Bitte wählen:")
        answer = input("Links/Rechts? ")
        if answer not in ["Links", "Rechts"]:
            answer = "Links"
        white_side = answer
        print(f"Setze Weiß auf {white_side}")

    # 3) Video nochmal abspielen, um Züge zu erkennen
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    previous_player_turn = None
    previous_fen = STARTING_PLACEMENT if found_start_position else None
    color = 'w'  # wir fangen standardmäßig an: Weiß am Zug

    # Leeres PGN-Spiel
    game = chess.pgn.Game()
    node = game
    global_board = chess.Board()  # Standard-Start

    if previous_fen is None:
        # Falls wir die Startstellung gar nicht haben, aber der user "Links" oder "Rechts" gewählt hat:
        # => global_board = chess.Board() => previous_fen = global_board.fen().split()[0] = rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
        # Du kannst es auch so lassen, je nachdem wie du's brauchst
        global_board = chess.Board()
        previous_fen = global_board.fen().split(" ")[0]

    while True:
        frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break  # Video Ende

        current_frame_index = int(frame_pos)

        if current_frame_index % frame_interval != 0:
            continue

        # Schachuhr
        player_turn, _ = detect_player_turn(frame, visualize=False)

        # Wenn sich die Uhrposition von left->right (etc.) ändert und nicht 'hold'
        if player_turn is not None and player_turn != 'hold' and player_turn != previous_player_turn:
            print(f"[Frame {current_frame_index}] Uhr geändert: {previous_player_turn} -> {player_turn}")
            midpoints, labels, confidences = detect_pieces(frame, min_conf=0.7)
            if midpoints.shape[0] > 0:
                # Perspektive
                ones = np.ones((midpoints.shape[0], 1))
                mid_hom = np.hstack([midpoints, ones])
                transformed = M @ mid_hom.T
                transformed /= transformed[2, :]
                transformed = transformed[:2, :].T

                if white_side == "Rechts":
                    rotated = np.zeros_like(transformed)
                    rotated[:, 0] = 800 - transformed[:, 0]
                    rotated[:, 1] = 800 - transformed[:, 1]
                    rotated_labels = labels[:]
                    rotated_conf = confidences[:]
                else:
                    rotated = transformed
                    rotated_labels = labels[:]
                    rotated_conf = confidences[:]

                current_fen = generate_placement_from_board(rotated, rotated_labels)

                if previous_fen is not None and current_fen != previous_fen:
                    print(f"Neue FEN: {current_fen}")
                    # Visualisierung
                    warped_image, _ = warp_perspective(frame, sorted_pts)
                    fig = plot_final_board(warped_image, rotated, rotated_labels, rotated_conf)
                    if fig:
                        plt.show()

                    move = fen_diff_to_move(previous_fen, current_fen, color=color)
                    if move and move in global_board.legal_moves:
                        # Legal
                        global_board.push(move)
                        node = node.add_variation(move)
                        print(f"Legal: {move} (color={color}) => Board: {global_board.fen()}")
                        # Farbe toggeln
                        color = 'b' if color == 'w' else 'w'
                        previous_fen = current_fen
                    else:
                        # Illegal => Fix: Frame-für-Frame
                        print("Kein legaler Zug => Versuche Frame-für-Frame-Fix ...")
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
                                print(f"Erfolgreich repariert: {move2} (color={color})")
                                color = 'b' if color == 'w' else 'w'
                                previous_fen = fixed_fen
                            else:
                                print("Trotz Fix kein legaler Zug -> FEN ignoriert.")
                        else:
                            print("Konnte keinen legalen Zug aus den Folgeframes ermitteln -> FEN ignoriert.")
                else:
                    if previous_fen is None:
                        previous_fen = current_fen

        previous_player_turn = player_turn

    cap.release()

    print("\n=== Resultierendes PGN ===")
    print(game)
    print("Fertig.")


if __name__ == "__main__":
    main()
