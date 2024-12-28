import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import chess
import chess.pgn
import chess.svg
import logging
import tempfile
import requests
from io import BytesIO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ==============================
# Konfiguration der Seite
# ==============================
st.set_page_config(page_title="CaroCam - Schach-Tracker", layout="wide")

# ------------------------------
# Dictionary für FEN-Symbole
# ------------------------------
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

STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
PERCENT_AB = 0.17
PERCENT_BC = -0.07

# -----------------------------
# Auffälligere Farben
# für jede Figurenklasse (BGR)
# -----------------------------
PIECE_COLORS = {
    'White Pawn':   (255, 255, 0),    # knalliges Gelb
    'White Knight': (255, 153, 51),   # Orange
    'White Bishop': (0, 255, 255),    # Cyan
    'White Rook':   (0, 255, 0),      # Hellgrün
    'White Queen':  (255, 0, 255),    # Magenta
    'White King':   (255, 128, 128),  # Hellrot
    'Black Pawn':   (128, 128, 128),  # Grau
    'Black Knight': (0, 0, 255),      # Rot
    'Black Bishop': (102, 0, 204),    # Violett
    'Black Rook':   (255, 0, 0),      # Blau im BGR
    'Black Queen':  (255, 255, 255),        # Schwarz
    'Black King':   (128, 0, 128)     # Dunkel-Lila
}

# ==============================
# Stockfish-Logik
# ==============================
def check_stockfish_api_available():
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    url = "https://stockfish.online/api/s/v2.php"
    params = {"fen": test_fen, "depth": 1}
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("success", False)
    except:
        pass
    return False

def analyze_fen_with_stockfish(fen, depth=15):
    url = 'https://stockfish.online/api/s/v2.php'
    params = {'fen': fen, 'depth': min(depth, 15)}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                raw_best_move = data.get("bestmove", "None")
                evaluation = data.get("evaluation", "None")
                mate = data.get("mate", None)

                best_move = None
                if raw_best_move != "None":
                    tokens = raw_best_move.split()
                    if len(tokens) >= 2:
                        best_move = tokens[1]
                return best_move, evaluation, mate
            else:
                return None, "None", None
        else:
            return None, "None", None
    except:
        return None, "None", None

def create_chessboard_svg(fen_str, last_move=None, check_color='red', highlight_color='green', size=350):
    board = chess.Board(fen_str)
    style = f"""
    .square.lastmove {{
        fill: {highlight_color} !important;
    }}
    .square.check {{
        fill: {check_color} !important;
    }}
    """
    lm = None
    if last_move:
        try:
            lm = chess.Move.from_uci(str(last_move))
        except:
            lm = None

    check_square = None
    if board.is_check():
        king_square = board.king(board.turn)
        if king_square is not None:
            check_square = king_square

    board_svg = chess.svg.board(
        board=board,
        lastmove=lm,
        check=check_square,
        size=size,
        style=style
    )
    return board_svg

def create_chessboard_svg_with_bestmove(fen_str, best_move_uci, arrow_color='blue', size=350):
    board = chess.Board(fen_str)
    arrows = []
    if best_move_uci:
        try:
            bm = chess.Move.from_uci(best_move_uci)
            arrows = [chess.svg.Arrow(bm.from_square, bm.to_square, color=arrow_color)]
        except:
            pass
    board_svg = chess.svg.board(board=board, size=size, arrows=arrows)
    return board_svg

# ==============================
# Model-Ladefunktionen
# ==============================
@st.cache_data(show_spinner=True)
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    piece_model_path = os.path.join(
        base_dir, 'Figure Detection 3', 'runs', 'detect', 'yolov8-chess32', 'weights', 'best.pt'
    )
    corner_model_path = os.path.join(
        base_dir, 'Corner Detection 2', 'runs', 'detect', 'yolov8n_corner14', 'weights', 'best.pt'
    )
    clock_model_path = os.path.join(
        base_dir, 'Clock Detection 3', 'runs', 'detect', 'yolov8-chess6', 'weights', 'best.pt'
    )

    if not (os.path.exists(piece_model_path)
            and os.path.exists(corner_model_path)
            and os.path.exists(clock_model_path)):
        st.error("Fehler: Modelldateien nicht gefunden. Bitte Pfade prüfen.")
        st.stop()

    piece_model = YOLO(piece_model_path)
    corner_model = YOLO(corner_model_path)
    clock_model = YOLO(clock_model_path)
    return piece_model, corner_model, clock_model

# ==============================
# Hilfsfunktionen
# ==============================
def detect_pieces(image, piece_model, min_conf=0.7):
    results = piece_model.predict(
        image,
        conf=0.05,
        iou=0.3,
        imgsz=1400,
        verbose=False
    )
    class_names = piece_model.model.names
    result = results[0]

    midpoints, labels, confidences = [], [], []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls.cpu().numpy()[0])
        conf_val = float(box.conf.cpu().numpy()[0])
        label = class_names[cls_id]

        if conf_val < min_conf:
            continue

        mid_x = x1 + (x2 - x1) / 2
        mid_y = y1 + (y2 - y1) * 0.75

        midpoints.append([mid_x, mid_y])
        labels.append(label)
        confidences.append(conf_val)
    return np.array(midpoints), labels, confidences, result

def detect_corners(image, corner_model):
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
        center_x = int((x1 + x2)/2)
        center_y = int((y1 + y2)/2)
        cls_id = int(box.cls.cpu().numpy()[0])
        if cls_id == 0:
            points["A"] = np.array([center_x, center_y])
        elif cls_id == 1:
            points["B"] = np.array([center_x, center_y])
        elif cls_id == 2:
            points["C"] = np.array([center_x, center_y])

    if len(points) != 3:
        return None, None
    return points, results

def detect_player_turn(image, clock_model):
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
    return player_turn, results[0]

def calculate_point_D(A, B, C):
    BC = C - B
    return A + BC

def adjust_point_D(A, B, C, D_calculated, percent_AB, percent_BC):
    AB = B - A
    BC = C - B
    correction_vector = percent_AB * AB + percent_BC * BC
    return D_calculated + correction_vector

def sort_points(A, B, C, D):
    pts = np.array([A, B, C, D])
    pts = sorted(pts, key=lambda x: x[1])
    top_pts = sorted(pts[:2], key=lambda x: x[0])
    bot_pts = sorted(pts[2:], key=lambda x: x[0])
    A_sorted, D_sorted = top_pts
    B_sorted, C_sorted = bot_pts
    return np.array([A_sorted, B_sorted, C_sorted, D_sorted], dtype=np.float32)

def warp_perspective(image, src_points):
    dst_size = 800
    dst_points = np.array([
        [0, 0],
        [0, dst_size - 1],
        [dst_size - 1, dst_size - 1],
        [dst_size - 1, 0]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, (dst_size, dst_size))
    # Brett drehen
    warped = cv2.rotate(warped, cv2.ROTATE_180)
    return warped, M

def generate_placement_from_board(midpoints, labels, grid_size=8):
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

# Fix-Funktion mit Rückgabe des passenden Frames
def fix_fen_with_single_frames_on_the_fly(
    cap, current_frame_index, frame_interval, M, white_side,
    old_fen, color, piece_model, max_tries=10, min_conf=0.7
):
    start_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    success_fen = None
    success_frame = None

    for _ in range(max_tries):
        next_frame_pos = current_frame_index + frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_pos)
        current_frame_index = next_frame_pos

        ret, frame = cap.read()
        if not ret:
            break

        midpoints, labels, confs, _ = detect_pieces(frame, piece_model, min_conf=min_conf)
        if midpoints.shape[0] == 0:
            continue

        ones = np.ones((midpoints.shape[0], 1))
        mid_hom = np.hstack([midpoints, ones])
        transformed = M @ mid_hom.T
        transformed /= transformed[2, :]
        transformed = transformed[:2, :].T

        if white_side == "Rechts":
            rotated = np.zeros_like(transformed)
            rotated[:, 0] = 800 - transformed[:, 0]
            rotated[:, 1] = 800 - transformed[:, 1]
        else:
            rotated = transformed

        new_fen = generate_placement_from_board(rotated, labels)
        move = fen_diff_to_move(old_fen, new_fen, color=color)
        if move:
            success_fen = new_fen
            success_frame = frame
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

    if success_fen is not None:
        return True, success_fen, success_frame
    else:
        return False, old_fen, None

def remove_pgn_headers(pgn_string):
    lines = pgn_string.split("\n")
    cleaned_lines = []
    for line in lines:
        if line.startswith("[") or line.strip() == "":
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# ===============================
# FUNKTIONEN FÜR DIE 2 BILDER
# ===============================
def create_mega_image(
        original,
        corners,
        D_calc,
        D_corr,
        piece_result,
        piece_midpoints,
        piece_labels,
        piece_confs,
        clock_boxes=None,
        corner_result=None
    ):
    """
    Bild 1:
      - Alle Ecken + D-Korrektur + Pfeil
      - Figuren-BoundingBoxes (eingefärbt nach Klassen) + Conf
      - Uhr (optional)
      - Mittelpunkte
      - Corner-BoundingBoxes (optional)
    """
    out = original.copy()

    # 1) Corner-BoundingBoxes
    if corner_result is not None:
        for box in corner_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls.cpu().numpy()[0])
            corner_label = corner_result.names[cls_id]
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            cv2.putText(
                out, corner_label,
                (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2
            )

    # 2) Ecken-Kreise + Beschriftung
    if corners is not None:
        A = corners["A"]
        B = corners["B"]
        C = corners["C"]
        cv2.circle(out, (A[0], A[1]), 10, (0, 0, 255), -1)
        cv2.putText(out, "A", (A[0]+10, A[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(out, (B[0], B[1]), 10, (0, 255, 255), -1)
        cv2.putText(out, "B", (B[0]+10, B[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.circle(out, (C[0], C[1]), 10, (255, 0, 255), -1)
        cv2.putText(out, "C", (C[0]+10, C[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # D calc/corr
        Dc = (int(D_calc[0]), int(D_calc[1]))
        Dco = (int(D_corr[0]), int(D_corr[1]))
        cv2.circle(out, Dc, 10, (255, 0, 0), -1)
        cv2.putText(out, "D_calc", (Dc[0]+10, Dc[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.circle(out, Dco, 10, (255, 0, 255), -1)
        cv2.putText(out, "D_corr", (Dco[0]+10, Dco[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Pfeil von D_calc zu D_corr
        cv2.arrowedLine(out, Dc, Dco, (255, 0, 255), 3)

    # 3) Figuren-BoundingBoxes
    for box in piece_result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls.cpu().numpy()[0])
        label = piece_result.names[cls_id]
        conf_val = float(box.conf.cpu().numpy()[0])
        color = PIECE_COLORS.get(label, (128, 128, 128))

        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        txt = f"{label} {conf_val*100:.1f}%"
        cv2.putText(out, txt, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 4) Mittelpunkte
    for (mx, my), lbl in zip(piece_midpoints, piece_labels):
        color = PIECE_COLORS.get(lbl, (0, 255, 0))
        cv2.circle(out, (int(mx), int(my)), 4, color, -1)

    # 5) Uhr
    if clock_boxes is not None:
        for box in clock_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.putText(out, "clock", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return out

def create_warped_image_with_grid_and_fen(warped, M, midpoints, labels):
    """
    Bild 2: Entzerrtes Brett mit Raster (ohne Beschriftung),
    Figuren-FEN-Kürzel an den Mittelpunkten.
    (nochmals um 180° gedreht zur Darstellung)
    """
    out = cv2.rotate(warped, cv2.ROTATE_180)

    # Raster
    cell = out.shape[0] // 8
    for i in range(9):
        # horizontale
        y = i * cell
        cv2.line(out, (0, y), (out.shape[1], y), (0, 0, 0), 2)
        # vertikale
        x = i * cell
        cv2.line(out, (x, 0), (x, out.shape[0]), (0, 0, 0), 2)

    # Transform midpoints
    ones = np.ones((midpoints.shape[0],1))
    hom = np.hstack([midpoints, ones])
    trans = M @ hom.T
    trans /= trans[2,:]
    trans = trans[:2,:].T

    for (mx, my), lbl in zip(trans, labels):
        fen_char = FEN_MAPPING.get(lbl, '?')
        px, py = int(mx), int(my)
        cv2.circle(out, (px, py), 4, (255, 0, 0), -1)
        cv2.putText(out, fen_char, (px+5, py+5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    return out

# =====================================
# Streamlit - Hauptfunktion
# =====================================
def main():
    st.title("Digitales Schachbrett-Tracking mit YOLO und python-chess")

    stockfish_available = check_stockfish_api_available()
    if stockfish_available:
        st.info("Stockfish-API ist verfügbar.")
    else:
        st.warning("Stockfish-API ist NICHT verfügbar.")

    piece_model, corner_model, clock_model = load_models()

    st.sidebar.header("Video-Upload")
    video_file = st.sidebar.file_uploader("Bitte ein Schachvideo hochladen", type=["mp4", "mov", "avi"])
    start_button = st.sidebar.button("Erkennung starten")

    frame_interval_factor = 0.2
    max_tries_fix = 60

    if not video_file:
        st.info("Bitte lade ein Video hoch, um zu starten.")
        return

    # Temporär speichern
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        temp_video_path = tmp.name
        tmp.write(video_file.read())

    if start_button:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Konnte das Video nicht öffnen.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        frame_interval = max(1, int(fps * frame_interval_factor))

        # ------------------------------------------------
        # Erstelle 2 Bilder aus dem allerersten Frame
        # ------------------------------------------------
        with st.expander("Zwei Bilder (1. Frame)"):
            st.write("Wir nehmen den ersten Frame und erstellen 2 Bilder:")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame = cap.read()
            if not ret:
                st.warning("Konnte keinen Frame lesen.")
            else:
                # 1) Erkennung Ecken + D
                cpoints, corner_res = detect_corners(first_frame, corner_model)
                if not cpoints:
                    st.warning("Keine Ecken erkannt (A,B,C).")
                else:
                    A = cpoints["A"]
                    B = cpoints["B"]
                    C = cpoints["C"]
                    D_calc = calculate_point_D(A,B,C)
                    D_corr = adjust_point_D(A,B,C,D_calc, PERCENT_AB, PERCENT_BC)

                # 2) Figuren
                midpoints, labels, confs, piece_res = detect_pieces(first_frame, piece_model)

                # 3) Uhr
                clock_label, clock_res = detect_player_turn(first_frame, clock_model)
                if clock_res and len(clock_res.boxes) > 0:
                    clock_boxes = clock_res.boxes
                else:
                    clock_boxes = None

                # Bild 1: Mega-Image
                if cpoints:
                    mega_img = create_mega_image(
                        original=first_frame,
                        corners=cpoints,
                        D_calc=D_calc,
                        D_corr=D_corr,
                        piece_result=piece_res,
                        piece_midpoints=midpoints,
                        piece_labels=labels,
                        piece_confs=confs,
                        clock_boxes=clock_boxes,
                        corner_result=corner_res[0]
                    )
                    st.subheader("Bild 1: Mega-Image mit ALLEM (Ecken, D, Figuren, Uhr, Mittelpunkte, Corner-Boxes)")
                    st.image(mega_img, channels="BGR")
                else:
                    st.warning("Konnte kein Mega-Image erzeugen, da Ecken fehlen.")

                # Bild 2: Entzerrtes Brett mit Raster
                if cpoints:
                    sorted_pts = sort_points(A,B,C,D_corr.astype(int))
                    warped, M_ = warp_perspective(first_frame, sorted_pts)
                    warped_img = create_warped_image_with_grid_and_fen(warped, M_, midpoints, labels)
                    st.subheader("Bild 2: Entzerrtes Brett (erneut um 180° gedreht) mit Gitter + FEN-Kürzeln")
                    st.image(warped_img, channels="BGR")
                else:
                    st.warning("Konnte kein entzerrtes Bild erzeugen, da Ecken fehlen.")

        # -------------------------------------------
        # Normales Prozedere (Zug-Tracking)
        # -------------------------------------------
        st.write("Starte das eigentliche Zug-Tracking...")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        corners_found = False
        M = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cpoints, _ = detect_corners(frame, corner_model)
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
                st.success("Ecken erkannt. Transformation bereit.")
                break

        if not corners_found or M is None:
            st.warning("Keine Brett-Ecken erkannt. Abbruch.")
            cap.release()
            return

        # Grundstellung
        st.write("Suche nach Grundstellung...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        white_side = None
        found_start_position = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            midpoints, labels, confs, _ = detect_pieces(frame, piece_model, min_conf=0.7)
            if midpoints.shape[0] == 0:
                continue

            ones = np.ones((midpoints.shape[0], 1))
            mid_hom = np.hstack([midpoints, ones])
            trans = M @ mid_hom.T
            trans /= trans[2, :]
            trans = trans[:2, :].T

            for side in ["Links", "Rechts"]:
                test_mid = np.copy(trans)
                if side == "Rechts":
                    test_mid[:, 0] = 800 - trans[:, 0]
                    test_mid[:, 1] = 800 - trans[:, 1]
                test_fen = generate_placement_from_board(test_mid, labels)
                if test_fen == STARTING_PLACEMENT:
                    white_side = side
                    found_start_position = True
                    st.success(f"Grundstellung erkannt: Weiß spielt auf {white_side}")
                    break

            if found_start_position:
                break

        if not found_start_position:
            st.warning("Keine automatische Grundstellung. Manuell wählen:")
            user_side = st.radio("Wo ist Weiß?", ["Links", "Rechts"])
            white_side = user_side
            st.info(f"Setze Weiß auf {white_side}.")

        st.write("Starte Zugerkennung...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        previous_player_turn = None
        if found_start_position:
            previous_fen = STARTING_PLACEMENT
        else:
            dummy_board = chess.Board()
            previous_fen = dummy_board.fen().split(" ")[0]

        color = 'w'
        game = chess.pgn.Game()
        node = game
        global_board = chess.Board()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        # Anstatt move_number => halfmove_count
        halfmove_count = 1

        while True:
            frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            if not ret:
                break

            current_progress = frame_pos / (total_frames if total_frames > 0 else 1)
            progress_bar.progress(min(current_progress, 1.0))

            if int(frame_pos) % frame_interval != 0:
                continue

            pturn, _clockresult = detect_player_turn(frame, clock_model)
            if pturn is not None and pturn != 'hold' and pturn != previous_player_turn:
                midpoints, labels, confs, _ = detect_pieces(frame, piece_model, min_conf=0.7)
                if midpoints.shape[0] > 0:
                    ones = np.ones((midpoints.shape[0], 1))
                    mid_hom = np.hstack([midpoints, ones])
                    trans = M @ mid_hom.T
                    trans /= trans[2, :]
                    trans = trans[:2, :].T

                    if white_side == "Rechts":
                        rotated = np.zeros_like(trans)
                        rotated[:, 0] = 800 - trans[:, 0]
                        rotated[:, 1] = 800 - trans[:, 1]
                        rotated_labels = labels[:]
                    else:
                        rotated = trans
                        rotated_labels = labels[:]

                    current_fen = generate_placement_from_board(rotated, rotated_labels)
                    move = fen_diff_to_move(previous_fen, current_fen, color=color)
                    if move and move in global_board.legal_moves:
                        global_board.push(move)
                        node = node.add_variation(move)

                        # Zug-Beschriftung: 1a oder 1b, 2a oder 2b ...
                        if color == 'w':
                            move_label = f"{(halfmove_count+1)//2}a: {move}"
                        else:
                            move_label = f"{(halfmove_count+1)//2}b: {move}"

                        with st.expander(move_label):
                            colL, colR = st.columns(2)
                            with colL:
                                st.image(frame, caption=f"Frame {frame_pos} - Original", channels="BGR")
                            with colR:
                                full_fen_current = f"{current_fen} {('w' if color=='b' else 'b')} KQkq - 0 1"
                                board_svg = create_chessboard_svg(full_fen_current, last_move=move)
                                st.markdown("**Vorheriger Zug**", unsafe_allow_html=True)
                                st.write(f"**FEN**: {current_fen}")
                                st.components.v1.html(board_svg, height=400)

                                if stockfish_available:
                                    key_ana = f"analysis_{full_fen_current}_{halfmove_count}"
                                    if key_ana not in st.session_state:
                                        st.session_state[key_ana] = analyze_fen_with_stockfish(full_fen_current)

                                    best_move, evaluation, mate = st.session_state[key_ana]
                                    if mate is not None and mate != 0:
                                        st.info(f"**Matt in {mate} Zügen!**")
                                    elif evaluation != "None":
                                        try:
                                            eval_val = float(evaluation)
                                            eval_clamped = max(-10, min(10, eval_val))
                                            percent = (eval_clamped + 10) / 20
                                            bar_html = f"""
                                            <div style='
                                                width: 300px; 
                                                height: 50px; 
                                                position: relative; 
                                                border: 2px solid #333; 
                                                background: linear-gradient(to right, white {percent*100}%, black {percent*100}% 100%);
                                                margin-top: 10px;
                                            '>
                                              <div style='
                                                position: absolute; 
                                                left: 50%; 
                                                top: 50%; 
                                                transform: translate(-50%, -50%);
                                                color: #aaa;
                                                font-weight: bold;
                                                font-size: 1.1em;
                                              '>{eval_val:.2f}</div>
                                            </div>
                                            """
                                            st.markdown(bar_html, unsafe_allow_html=True)
                                        except:
                                            pass

                                    if best_move:
                                        st.write(f"**Empfohlener nächster Zug:** {best_move}")
                                        board_svg_bm = create_chessboard_svg_with_bestmove(full_fen_current, best_move)
                                        st.components.v1.html(board_svg_bm, height=400)

                        # Farbe wechseln
                        color = 'b' if color == 'w' else 'w'
                        previous_fen = current_fen
                        halfmove_count += 1

                    else:
                        # Korrektur
                        fixed_ok, fixed_fen, fix_frame = fix_fen_with_single_frames_on_the_fly(
                            cap=cap,
                            current_frame_index=int(frame_pos),
                            frame_interval=frame_interval,
                            M=M,
                            white_side=white_side,
                            old_fen=previous_fen,
                            color=color,
                            piece_model=piece_model,
                            max_tries=max_tries_fix,
                            min_conf=0.7
                        )
                        if fixed_ok:
                            move2 = fen_diff_to_move(previous_fen, fixed_fen, color=color)
                            if move2 and move2 in global_board.legal_moves:
                                global_board.push(move2)
                                node = node.add_variation(move2)

                                # Zug-Beschriftung
                                if color == 'w':
                                    move_label2 = f"{(halfmove_count+1)//2}a: {move2} (Korrektur)"
                                else:
                                    move_label2 = f"{(halfmove_count+1)//2}b: {move2} (Korrektur)"

                                with st.expander(move_label2):
                                    colL, colR = st.columns(2)
                                    with colL:
                                        st.image(
                                            fix_frame,
                                            caption=f"Korrektur-Frame mit gültigem Zug",
                                            channels="BGR"
                                        )
                                    with colR:
                                        full_fen_fixed = f"{fixed_fen} {('w' if color=='b' else 'b')} KQkq - 0 1"
                                        board_svg = create_chessboard_svg(full_fen_fixed, last_move=move2)
                                        st.markdown("<h3>Vorheriger Zug</h3>", unsafe_allow_html=True)
                                        st.write(f"**FEN**: {fixed_fen}")
                                        st.components.v1.html(board_svg, height=400)

                                        if stockfish_available:
                                            kfix = f"analysis_{full_fen_fixed}_{halfmove_count}"
                                            if kfix not in st.session_state:
                                                st.session_state[kfix] = analyze_fen_with_stockfish(full_fen_fixed)

                                            bm2, ev2, mt2 = st.session_state[kfix]
                                            if mt2 is not None and mt2 != 0:
                                                st.info(f"**Matt in {mt2} Zügen!**")
                                            elif ev2 != "None":
                                                try:
                                                    val2 = float(ev2)
                                                    clamp2 = max(-10, min(10, val2))
                                                    p2 = (clamp2 + 10) / 20
                                                    bar_html2 = f"""
                                                    <div style='
                                                        width: 300px; 
                                                        height: 50px; 
                                                        position: relative; 
                                                        border: 2px solid #333; 
                                                        background: linear-gradient(to right, white {p2*100}%, black {p2*100}% 100%);
                                                        margin-top: 10px;
                                                    '>
                                                      <div style='
                                                        position: absolute; 
                                                        left: 50%; 
                                                        top: 50%; 
                                                        transform: translate(-50%, -50%);
                                                        color: #aaa;
                                                        font-weight: bold;
                                                        font-size: 1.1em;
                                                      '>{val2:.2f}</div>
                                                    </div>
                                                    """
                                                    st.markdown(bar_html2, unsafe_allow_html=True)
                                                except:
                                                    pass

                                            if bm2:
                                                st.write(f"**Empfohlener nächster Zug:** {bm2}")
                                                svg_bm2 = create_chessboard_svg_with_bestmove(full_fen_fixed, bm2)
                                                st.components.v1.html(svg_bm2, height=400)

                                color = 'b' if color == 'w' else 'w'
                                previous_fen = fixed_fen
                                halfmove_count += 1
                            else:
                                pass
                        else:
                            pass
                else:
                    pass

            previous_player_turn = pturn
            if global_board.is_game_over():
                break

        cap.release()
        progress_bar.empty()

        if global_board.is_game_over():
            if global_board.is_checkmate():
                winner = "Weiß" if color == 'b' else "Schwarz"
                st.success(f"Glückwunsch, {winner} hat Matt gesetzt!")
            else:
                st.info("Partie ist zu Ende (z.B. Patt oder Abbruch).")
        else:
            st.info("Video zu Ende, keine weiteren Züge erkannt.")

        raw_pgn = str(game)
        moves_only_pgn = remove_pgn_headers(raw_pgn)

        st.markdown(
            """<h2 style='text-align:center; font-size:1.6em; 
                         color:#b00; margin-top:40px;'>
               Abschließende Partienotation
               </h2>""",
            unsafe_allow_html=True
        )
        st.code(moves_only_pgn, language="plaintext")
        st.balloons()


if __name__ == "__main__":
    main()
