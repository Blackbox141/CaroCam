import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import chess
import chess.pgn
import logging
import tempfile
import time

# Logging für ultralytics unterdrücken
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ---------------------------------------------------------------
# Globale Konstanten & Mappings
# ---------------------------------------------------------------

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

# ---------------------------------------------------------------
# Laden bzw. Initialisieren der YOLO-Modelle
# (Beim ersten Aufruf kann das etwas dauern)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_models():
    # Passe hier die Pfade an deine lokale Struktur an:
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


# ---------------------------------------------------------------
# Unterstützende Funktionen
# ---------------------------------------------------------------
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
    return np.array(midpoints), labels, confidences


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


def detect_player_turn(image, clock_model, visualize=False):
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


def plot_final_board(image, midpoints, labels, confidences):
    import matplotlib.pyplot as plt
    from io import BytesIO

    if midpoints.shape[0] == 0:
        return None

    grid_size = 8
    step_size = image.shape[0] // grid_size

    fig, ax = plt.subplots(figsize=(6, 6))
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

    for (x, y), lbl, conf_val in zip(midpoints, labels, confidences):
        col = int(x // step_size)
        row = int(y // step_size)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            fen_char = FEN_MAPPING.get(lbl, '?')
            cx = col * step_size + step_size / 2
            cy = row * step_size + step_size / 2
            text_str = f"{fen_char}({conf_val*100:.1f}%)"
            ax.text(cx, cy, text_str, fontsize=14, color='red',
                    ha='center', va='center')

    ax.set_title("Erkannte Figuren (Confidence)", fontsize=12)
    ax.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def fix_fen_with_single_frames_on_the_fly(cap, current_frame_index, frame_interval,
                                          M, white_side,
                                          old_fen, color,
                                          piece_model,
                                          max_tries=10,
                                          min_conf=0.7):
    """
    Liest nacheinander bis zu max_tries Frames -> Figurenerkennung -> FEN,
    prüft, ob fen_diff_to_move(old_fen, new_fen) legal. Falls ja, return.
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

        midpoints, labels, confidences = detect_pieces(frame, piece_model, min_conf=min_conf)
        if midpoints.shape[0] == 0:
            continue

        # Perspektivisch transformieren
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

    # Zurücksetzen
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

    if success_fen is not None:
        return True, success_fen
    else:
        return False, old_fen


# ---------------------------------------------------------------
# STREAMLIT-HAUPTANWENDUNG
# ---------------------------------------------------------------
def main():
    st.title("Digitales Schachbrett-Tracking mit YOLO und python-chess")
    st.write("""
    Diese Anwendung lädt ein Schach-Video hoch, erkennt das Brett und die Figuren,
    leitet daraus Züge ab und zeigt am Ende das PGN an.
    """)

    piece_model, corner_model, clock_model = load_models()

    # --- Seitenleiste für Upload und Parameter ---
    st.sidebar.header("Video-Upload & Einstellungen")
    video_file = st.sidebar.file_uploader("Bitte ein Schachvideo hochladen", type=["mp4", "mov", "avi"])
    frame_interval_factor = st.sidebar.slider("Frame-Intervall (Sek.)", 0.1, 1.0, 0.2, 0.1)
    max_tries_fix = st.sidebar.number_input("Max. Frame-Fix-Versuche", 1, 50, 10)

    start_button = st.sidebar.button("Erkennung starten")

    if not video_file:
        st.info("Bitte lade ein Video hoch, um zu starten.")
        return

    # Wir speichern das Video im temporären Ordner, damit OpenCV es lesen kann
    temp_video_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        temp_video_path = tmp.name
        tmp.write(video_file.read())

    if start_button:
        # Hier beginnt die Analyse
        st.write("Starte Videoanalyse...")
        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            st.error("Konnte das Video nicht öffnen.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        frame_interval = max(1, int(fps * frame_interval_factor))

        # Schritt 1: Ecken finden
        corners_found = False
        M = None

        with st.spinner("Suche nach Brett-Ecken..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cpoints = detect_corners(frame, corner_model)
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
                    st.success("Ecken gefunden. Perspektivische Transformation vorbereitet.")
                    break

        if not corners_found or M is None:
            st.warning("Keine Brett-Ecken erkannt. Abbruch.")
            cap.release()
            return

        # Suche nach Grundstellung -> Wo ist Weiß?
        st.write("Suche nach Grundstellung (Standard-Aufstellung), um zu erkennen, ob Weiß links oder rechts spielt.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        white_side = None
        found_start_position = False

        while True:
            ret, frame = cap.read()
            if not ret:
                # Video Ende -> nicht gefunden
                break
            midpoints, labels, _ = detect_pieces(frame, piece_model, min_conf=0.7)
            if midpoints.shape[0] == 0:
                continue

            ones = np.ones((midpoints.shape[0], 1))
            mid_hom = np.hstack([midpoints, ones])
            transformed = M @ mid_hom.T
            transformed /= transformed[2, :]
            transformed = transformed[:2, :].T

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
                    found_start_position = True
                    st.success(f"Grundstellung erkannt -> Weiß spielt auf {white_side}")
                    break
            if found_start_position:
                break

        if not found_start_position:
            st.warning("Konnte Grundstellung nicht automatisch erkennen.")
            user_side = st.radio("Bitte wählen, wo Weiß ist:", ["Links", "Rechts"])
            white_side = user_side
            st.info(f"Setze Weiß auf {white_side}.")

        # Hauptloop: Erkennen von Zügen
        st.write("Starte nun die Zugerkennung...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        previous_player_turn = None
        if found_start_position:
            previous_fen = STARTING_PLACEMENT
        else:
            # Falls nicht gefunden, Standard-Board
            global_board_temp = chess.Board()
            previous_fen = global_board_temp.fen().split(" ")[0]

        color = 'w'

        game = chess.pgn.Game()
        node = game
        global_board = chess.Board()  # Startstellung

        # Fortschrittsanzeige
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        while True:
            frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            if not ret:
                break

            current_progress = frame_pos / (total_frames if total_frames > 0 else 1)
            progress_bar.progress(min(current_progress, 1.0))

            if int(frame_pos) % frame_interval != 0:
                continue

            player_turn, _ = detect_player_turn(frame, clock_model, visualize=False)
            if player_turn is not None and player_turn != 'hold' and player_turn != previous_player_turn:
                st.write(f"Uhrwechsel erkannt (Frame {frame_pos}): {previous_player_turn} -> {player_turn}")
                midpoints, labels, confidences = detect_pieces(frame, piece_model, min_conf=0.7)
                if midpoints.shape[0] > 0:
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
                    if current_fen != previous_fen:
                        st.write(f"Neue FEN: {current_fen}")
                        # Visualisieren:
                        warped_image, _ = warp_perspective(frame, sorted_pts)
                        fig_buf = plot_final_board(warped_image, rotated, rotated_labels, rotated_conf)
                        if fig_buf:
                            st.image(fig_buf, caption=f"Frame {frame_pos}: erkannte Figuren")

                        move = fen_diff_to_move(previous_fen, current_fen, color=color)
                        if move and move in global_board.legal_moves:
                            global_board.push(move)
                            node = node.add_variation(move)
                            st.write(f"Legal: {move} (Farbe={color}) => Neuer Board-Status: {global_board.fen()}")
                            color = 'b' if color == 'w' else 'w'
                            previous_fen = current_fen
                        else:
                            st.info("Kein legaler Zug -> Versuche Frame-für-Frame-Korrektur...")
                            fixed_ok, fixed_fen = fix_fen_with_single_frames_on_the_fly(
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
                                    st.success(f"Erfolgreich repariert: {move2} (Farbe={color})")
                                    color = 'b' if color == 'w' else 'w'
                                    previous_fen = fixed_fen
                                else:
                                    st.warning("Obwohl repariert, noch immer kein legaler Zug. Ignoriere.")
                            else:
                                st.warning("Konnte nach mehreren Versuchen keinen legalen Zug finden. Ignoriere.")
                    else:
                        if previous_fen is None:
                            previous_fen = current_fen
                else:
                    st.write("Keine Figuren erkannt in diesem Frame.")
            previous_player_turn = player_turn

            # Prüfen, ob das Spiel vorbei ist (z. B. Matt oder Patt)
            if global_board.is_game_over():
                st.warning("Spiel ist beendet.")
                break

        cap.release()
        progress_bar.empty()

        # Spiel zu Ende oder Video fertig
        if global_board.is_game_over():
            # Schachmatt, Patt, Aufgabe etc.
            if global_board.is_checkmate():
                st.success("**Glückwunsch!** Schachmatt erkannt.")
            else:
                st.info("Partie ist zu Ende (Patt oder andere Endbedingung).")
        else:
            st.info("Video zu Ende, keine weiteren Züge erkannt.")

        # Ausgeben des PGN
        pgn_text = str(game)
        st.subheader("Ermitteltes PGN:")
        st.text_area("Partie", value=pgn_text, height=200)
        st.download_button("PGN herunterladen", pgn_text, file_name="partie.pgn", mime="text/plain")

        st.balloons()  # Kleiner Effekt am Ende


# Streamlit-Einstiegspunkt
if __name__ == "__main__":
    main()
