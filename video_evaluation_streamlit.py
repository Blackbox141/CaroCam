import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import os
import chess
import chess.pgn
import chess.svg
import logging
import tempfile
import requests
import time

# ==============================
# Logging Konfiguration
# ==============================
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ==============================
# EXAKTE UI: Wie im alten Code
# + deine neuen Wünsche
# ==============================
st.set_page_config(page_title="CaroCam - Schach-Tracker", layout="wide")
st.image("CaroCam Logo.png", use_container_width=False, width=450)

st.title("KI-gestützte Schach-Analyse")
st.markdown("**Eine Projektarbeit von Luca Lenherr und Dennis Langer**")

# ==============================
# Konfigurationsparameter
# ==============================
PERCENT_AB = 0.17
PERCENT_BC = -0.07

INTERVAL_NORMAL_SECONDS = 1.0   # Zustand 1 (Normal)
INTERVAL_SEARCH_SECONDS = 0.2   # Zustand 2 (Searching)
INTERVAL_FIX_SECONDS = 0.15     # Zustand 3 (Fixing)
MAX_FRAMES_FIX = 80            # maximale Frames im Fixierzustand
MIN_CONF_PIECE = 0.7           # Mindestkonfidenz
PADDING = 30                   # Padding für ROI

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

# Figuren-Labels mit Farben (für Bounding Boxes)
PIECE_COLORS = {
    'White Pawn':   (255, 255, 0),
    'White Knight': (255, 153, 51),
    'White Bishop': (0, 255, 255),
    'White Rook':   (0, 255, 0),
    'White Queen':  (255, 0, 255),
    'White King':   (255, 128, 128),
    'Black Pawn':   (128, 128, 128),
    'Black Knight': (0, 0, 255),
    'Black Bishop': (102, 0, 204),
    'Black Rook':   (255, 0, 0),
    'Black Queen':  (255, 255, 255),
    'Black King':   (128, 0, 128)
}

# YOLO-Configs
PIECE_DETECTION_CONFIG = {
    'min_conf': MIN_CONF_PIECE,
    'conf': 0.05,
    'iou': 0.3,
    'imgsz': 1400,
    'verbose': False
}
CORNER_DETECTION_CONFIG = {
    'conf': 0.1,
    'iou': 0.3,
    'imgsz': 1400,
    'verbose': False
}
CLOCK_DETECTION_CONFIG = {
    'conf': 0.5,
    'iou': 0.6,
    'imgsz': 1200,
    'verbose': False
}
CLOCK_DETECTION_ROI_CONFIG = {
    'conf': 0.5,
    'iou': 0.7,
    'imgsz': 700,
    'verbose': False
}

CLOCK_LABEL_SYNONYMS = {
    'links': 'left',
    'rechts': 'right',
    'leftclock': 'left',
    'rightclock': 'right'
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
        return None, "None", None
    except:
        return None, "None", None

# ==============================
# SVG-Erzeugung
# ==============================
import chess.svg

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
        ksq = board.king(board.turn)
        if ksq is not None:
            check_square = ksq

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
# YOLO-Modelle
# ==============================
@st.cache_resource(show_spinner=True)
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    piece_model_path = os.path.join(
        base_dir, 'Figure Detection Model', 'runs', 'detect', 'yolov8-chess32', 'weights', 'best.pt'
    )
    corner_model_path = os.path.join(
        base_dir, 'Corner Detection Model', 'runs', 'detect', 'yolov8n_corner14', 'weights', 'best.pt'
    )
    clock_model_path = os.path.join(
        base_dir, 'Clock Detection Model', 'runs', 'detect', 'yolov8-chess6', 'weights', 'best.pt'
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
def detect_pieces(image, piece_model, config):
    results = piece_model.predict(
        image,
        conf=config['conf'],
        iou=config['iou'],
        imgsz=config['imgsz'],
        verbose=config['verbose']
    )
    class_names = piece_model.model.names
    res0 = results[0]

    midpoints, labels, confidences = [], [], []
    for box in res0.boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        c_id= int(box.cls.cpu().numpy()[0])
        conf_val= float(box.conf.cpu().numpy()[0])
        label = class_names[c_id]

        if conf_val < config['min_conf']:
            continue

        mid_x = x1 + (x2 - x1) / 2
        mid_y = y1 + (y2 - y1) * 0.75  # kleiner Offset nach unten

        midpoints.append([mid_x, mid_y])
        labels.append(label)
        confidences.append(conf_val)
    return np.array(midpoints), labels, confidences, res0

def detect_corners(image, corner_model, config):
    results = corner_model.predict(
        image,
        conf=config['conf'],
        iou=config['iou'],
        imgsz=config['imgsz'],
        verbose=config['verbose']
    )
    points = {}
    for box in results[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        c_id= int(box.cls.cpu().numpy()[0])
        if c_id == 0:
            points["A"] = np.array([cx,cy])
        elif c_id == 1:
            points["B"] = np.array([cx,cy])
        elif c_id == 2:
            points["C"] = np.array([cx,cy])
    if len(points) != 3:
        return None, None
    return points, results[0]

def detect_clock(image, clock_model, config):
    results = clock_model.predict(
        image,
        conf=config['conf'],
        iou=config['iou'],
        imgsz=config['imgsz'],
        verbose=config['verbose']
    )
    c_info = []
    for box in results[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        cid= int(box.cls.cpu().numpy()[0])
        conf_= float(box.conf.cpu().numpy()[0])
        raw_label= clock_model.model.names[cid].lower()
        if raw_label in CLOCK_LABEL_SYNONYMS:
            raw_label = CLOCK_LABEL_SYNONYMS[raw_label]
        c_info.append({
            'bbox':[x1,y1,x2,y2],
            'label': raw_label,
            'confidence': conf_
        })
    return c_info

def calculate_point_D(A,B,C):
    BC = C - B
    return A + BC

def adjust_point_D(A,B,C, D_calc, pAB,pBC):
    AB = B - A
    BC = C - B
    corr = pAB * AB + pBC * BC
    return D_calc + corr

def sort_points(A,B,C,D):
    pts= np.array([A,B,C,D])
    pts= sorted(pts, key=lambda x:x[1])  # nach y sortieren
    top= sorted(pts[:2], key=lambda x:x[0])
    bot= sorted(pts[2:], key=lambda x:x[0])
    A_,D_ = top
    B_,C_ = bot
    return np.array([A_, B_, C_, D_], dtype=np.float32)

def warp_perspective(img, src_points):
    dst_size = 800
    dst_pts= np.array([
        [0,0],
        [0,dst_size-1],
        [dst_size-1,dst_size-1],
        [dst_size-1,0]
    ], dtype=np.float32)
    M= cv2.getPerspectiveTransform(src_points, dst_pts)
    warped= cv2.warpPerspective(img, M, (dst_size,dst_size))
    # Brett wird rotiert, damit A oben links liegt
    warped= cv2.rotate(warped, cv2.ROTATE_180)
    return warped, M

def generate_placement_from_board(transformed, labels, grid_size=8):
    board = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    step_size= 800 // grid_size
    for (mx,my), lbl in zip(transformed, labels):
        c= int(my // step_size)
        r= 7 - int(mx // step_size)
        if 0 <= r < 8 and 0 <= c < 8:
            fen_char= FEN_MAPPING.get(lbl,'')
            if fen_char:
                board[r][c] = fen_char

    fen_rows = []
    for row_ in board:
        empty_count= 0
        fen_row= ''
        for sq in row_:
            if sq=='':
                empty_count += 1
            else:
                if empty_count>0:
                    fen_row += str(empty_count)
                    empty_count= 0
                fen_row+= sq
        if empty_count>0:
            fen_row+= str(empty_count)
        fen_rows.append(fen_row)
    return '/'.join(fen_rows)

def fen_diff_to_move(fen1, fen2, color='w'):
    if not fen1 or not fen2:
        return None
    full_fen1= f"{fen1} {color} KQkq - 0 1"
    try:
        board= chess.Board(full_fen1)
    except:
        return None
    for mv in board.legal_moves:
        board.push(mv)
        fenp= board.fen().split(" ")[0]
        if fenp == fen2:
            return mv
        board.pop()
    return None

def remove_pgn_headers(pgn_string):
    lines= pgn_string.split("\n")
    cleaned=[]
    for line in lines:
        if line.startswith("[") or line.strip()=="":
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def expand_bbox(x1,y1,x2,y2,padding, fw,fh):
    x1n= max(0, int(x1)-padding)
    y1n= max(0, int(y1)-padding)
    x2n= min(int(fw), int(x2)+padding)
    y2n= min(int(fh), int(y2)+padding)
    return (x1n,y1n,x2n,y2n)

# ==============================
# Rückwärts-Suche
# (Einmaliges "Glückwunsch..." entfernt)
# ==============================
def search_backwards_for_mate_or_matt(cap, start_frame, max_frames_fix, piece_model, state_info, frame_interval_search):
    global_board= state_info['global_board']
    color= state_info['color']
    M_= state_info['M']
    w_side= state_info['white_side']
    prev_fen= state_info['previous_fen']
    node= state_info['node']
    halfmove_count= state_info['halfmove_count']

    current_frame= start_frame
    count_attempts= 0

    while current_frame>=0:
        if count_attempts> max_frames_fix:
            st.info("Rückwärtssuche abgebrochen: max_frames_fix erreicht.")
            break
        count_attempts+=1

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame_b= cap.read()
        if not ret:
            current_frame-= frame_interval_search
            continue

        # Figuren erkennen
        midb, labsb, confsb, _= detect_pieces(frame_b, piece_model, PIECE_DETECTION_CONFIG)
        if midb.shape[0]==0:
            current_frame-= frame_interval_search
            continue

        onesb= np.ones((midb.shape[0],1))
        mhb= np.hstack([midb, onesb])
        transfb= M_ @ mhb.T
        transfb/= transfb[2,:]
        transfb= transfb[:2,:].T

        if w_side=='rechts':
            rotb= np.zeros_like(transfb)
            rotb[:,0]= 800 - transfb[:,0]
            rotb[:,1]= 800 - transfb[:,1]
        else:
            rotb= transfb

        fenb= generate_placement_from_board(rotb, labsb)
        mvb= fen_diff_to_move(prev_fen, fenb, color=color)
        if mvb and mvb in global_board.legal_moves:
            global_board.push(mvb)
            node= node.add_variation(mvb)

            with st.expander(f"Zug {(halfmove_count+1)//2}{'a' if color=='w' else 'b'}: {mvb}"):
                # Gewünschter Titel
                st.markdown("#### Gespielter Zug")
                colL, colR= st.columns(2)
                with colL:
                    st.image(frame_b, caption=f"Frame {current_frame} (Rückwärtssuche)", channels="BGR")
                with colR:
                    fen_for_board= f"{fenb} {('w' if color=='b' else 'b')} KQkq - 0 1"
                    bsvg= create_chessboard_svg(fen_for_board, last_move= mvb)
                    st.markdown("**Vorheriger Zug**", unsafe_allow_html=True)
                    st.write(f"**FEN**: {fenb}")
                    st.components.v1.html(bsvg, height=400)

                    if state_info['stockfish_available']:
                        keyA= f"analysis_{fen_for_board}_{halfmove_count}"
                        if keyA not in st.session_state:
                            st.session_state[keyA]= analyze_fen_with_stockfish(fen_for_board)
                        bestM,evv,mate= st.session_state[keyA]
                        if mate and mate!=0:
                            st.info(f"**Matt in {mate} Zügen**!")
                        elif evv!="None":
                            try:
                                e_val= float(evv)
                                e_cl= max(-10,min(10,e_val))
                                pr_= (e_cl+10)/20
                                bar_ = f"""
                                <div style='width:300px;height:50px;position:relative;border:2px solid #333;
                                background:linear-gradient(to right,white {pr_*100}%,black {pr_*100}% 100%);
                                margin-top:10px;'>
                                  <div style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                                  color:#aaa;font-weight:bold;font-size:1.1em;'>{e_val:.2f}</div>
                                </div>
                                """
                                st.markdown(bar_, unsafe_allow_html=True)
                            except:
                                pass
                        if bestM:
                            st.markdown("#### Empfohlener nächster Zug")
                            st.write(f"**Empfohlener nächster Zug**: {bestM}")
                            bsvg2= create_chessboard_svg_with_bestmove(fen_for_board,bestM)
                            st.components.v1.html(bsvg2, height=400)

            if global_board.is_game_over():
                state_info['state']="finished"
                return state_info

            state_info['color']= 'b' if color=='w' else 'w'
            state_info['previous_fen']= fenb
            state_info['halfmove_count']+=1

        current_frame-= frame_interval_search

    st.info("Rückwärtssuche beendet: kein Matt gefunden.")
    return state_info


# =====================================
# MAIN-Funktion (Zustandsmaschine etc.)
# =====================================
def main():
    # --- Stockfish-Verfügbarkeit checken
    stockfish_available = check_stockfish_api_available()
    if stockfish_available:
        st.success("Stockfish-API ist verfügbar.")
    else:
        st.warning("Stockfish-API ist NICHT verfügbar.")

    # --- Modelle laden
    piece_model, corner_model, clock_model= load_models()

    # --- Sidebar
    st.sidebar.header("Video-Upload")
    video_file= st.sidebar.file_uploader("Bitte ein Schachvideo hochladen", type=["mp4","mov","avi"])
    start_button= st.sidebar.button("Erkennung starten")

    if not video_file:
        st.info("Bitte lade ein Video hoch, um zu starten.")
        return

    # Sidebar: Fortschritts-Anzeige und Frame-Zähler
    progress_bar = st.sidebar.progress(0)
    frame_counter_placeholder = st.sidebar.empty()

    # --- Neues Game
    game= chess.pgn.Game()
    node= game
    global_board= chess.Board()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        temp_video_path = tmp.name
        tmp.write(video_file.read())

    if start_button:
        cap= cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Konnte das Video nicht öffnen.")
            return

        fps= cap.get(cv2.CAP_PROP_FPS)
        if fps<=0:
            fps=25.0

        frame_interval_normal= max(1, int(fps*INTERVAL_NORMAL_SECONDS))
        frame_interval_search= max(1, int(fps*INTERVAL_SEARCH_SECONDS))
        frame_interval_fix= max(1, int(fps*INTERVAL_FIX_SECONDS))
        total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # =========================
        # Brett-Ecken detektieren
        # =========================
        st.write("Starte Analyse...")
        st.write("Suche Brett-Ecken...")
        found_corners= False
        M_= None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame_test= cap.read()
            if not ret:
                break
            cpoints,_= detect_corners(frame_test, corner_model, CORNER_DETECTION_CONFIG)
            if cpoints:
                found_corners= True
                A= cpoints["A"]
                B= cpoints["B"]
                C= cpoints["C"]
                D_calc= calculate_point_D(A,B,C)
                D_corr= adjust_point_D(A,B,C, D_calc, PERCENT_AB, PERCENT_BC)
                sorted_pts= sort_points(A,B,C,D_corr.astype(int))
                M_= cv2.getPerspectiveTransform(
                    sorted_pts,
                    np.array([[0,0],[0,799],[799,799],[799,0]], dtype=np.float32)
                )
                st.success("Ecken erkannt.")
                break

        if not found_corners or M_ is None:
            st.error("Keine Brett-Ecken erkannt. Abbruch.")
            cap.release()
            os.remove(temp_video_path)
            return

        # =========================
        # Expander: Hinter den Kulissen
        # =========================
        with st.expander("Hinter den Kulissen"):
            st.write("Hier siehst du, wie das Schachbrett analysiert wird.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame= cap.read()
            if not ret:
                st.warning("Konnte keinen Frame lesen.")
            else:
                cpoints_dbg, corner_res = detect_corners(first_frame, corner_model, CORNER_DETECTION_CONFIG)
                midpoints, labels, confs, piece_res = detect_pieces(first_frame, piece_model, PIECE_DETECTION_CONFIG)
                cinfo_= detect_clock(first_frame, clock_model, CLOCK_DETECTION_CONFIG)

                preview= first_frame.copy()

                # 1) Eckpunkte + Bounding Box
                if corner_res and len(corner_res.boxes) > 0:
                    for box_ in corner_res.boxes:
                        x1, y1, x2, y2 = box_.xyxy[0].cpu().numpy().astype(int)
                        cx_ = (x1 + x2)//2
                        cy_ = (y1 + y2)//2
                        cid_ = int(box_.cls.cpu().numpy()[0])
                        corner_label = corner_model.model.names[cid_]

                        # Bounding Box (Pink)
                        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        # Mittelpunkt (andere Farbe, z.B. Gelb)
                        cv2.circle(preview, (cx_, cy_), 6, (0, 255, 255), -1)
                        # Text
                        cv2.putText(
                            preview, f"{corner_label} (Mittelpunkt)",
                            (x1, max(0, y2 + 20)),   # Unterhalb des BBox
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 255),
                            2
                        )

                # 2) Korrekturvektor: D_calc, D_corr
                if cpoints_dbg:
                    A_ = cpoints_dbg["A"]
                    B_ = cpoints_dbg["B"]
                    C_ = cpoints_dbg["C"]
                    D_calc_ = calculate_point_D(A_, B_, C_)
                    D_corr_ = adjust_point_D(A_, B_, C_, D_calc_, PERCENT_AB, PERCENT_BC)

                    # Pfeil von D_calc_ zu D_corr_
                    cv2.arrowedLine(
                        preview,
                        (int(D_calc_[0]), int(D_calc_[1])),
                        (int(D_corr_[0]), int(D_corr_[1])),
                        (0, 255, 0), 3
                    )
                    # Kreise + Labels
                    cv2.circle(preview, (int(D_calc_[0]), int(D_calc_[1])), 10, (0, 255, 255), -1)
                    cv2.putText(
                        preview, "D_calc",
                        (int(D_calc_[0] + 10), int(D_calc_[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )
                    cv2.circle(preview, (int(D_corr_[0]), int(D_corr_[1])), 10, (0, 0, 255), -1)
                    cv2.putText(
                        preview, "D_corr",
                        (int(D_corr_[0] + 10), int(D_corr_[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                # 3) Figuren-BoundingBoxes
                if piece_res and len(piece_res.boxes) > 0:
                    for box_ in piece_res.boxes:
                        x1, y1, x2, y2 = box_.xyxy[0].cpu().numpy().astype(int)
                        cid_ = int(box_.cls.cpu().numpy()[0])
                        p_label_ = piece_model.model.names[cid_]
                        confv_ = float(box_.conf.cpu().numpy()[0])
                        color_ = PIECE_COLORS.get(p_label_, (255, 0, 0))
                        cv2.rectangle(preview, (x1, y1), (x2, y2), color_, 2)
                        txt_ = f"{p_label_} {confv_*100:.1f}%"
                        cv2.putText(preview, txt_,
                                    (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    color_,
                                    2)

                # 4) Uhr => nur ein Label + ROI-Beschriftung unterhalb der Box
                if cinfo_:
                    best_clock = max(cinfo_, key=lambda c: c['confidence'])
                    x1c, y1c, x2c, y2c = map(int, best_clock['bbox'])
                    cv2.rectangle(preview, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
                    cv2.putText(preview, "clock",
                                (x1c, max(0, y2c + 20)),  # unterhalb der Box
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2)

                    # ROI-BoundingBox
                    fh, fw = first_frame.shape[:2]
                    roi_ = expand_bbox(x1c, y1c, x2c, y2c, PADDING, fw, fh)
                    rx1, ry1, rx2, ry2 = roi_
                    cv2.rectangle(preview, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                    cv2.putText(preview, "ROI der Uhr",
                                (rx1, ry2 + 20),  # unterhalb
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2)

                # Ausgabe
                st.subheader("Alle erkannten Objekte (Eckpunkte des Bretts, Figuren und Uhr, inkl. Korrekturvektor und ROI)")
                st.image(preview, channels="BGR")

                # 5) Entzerrtes Brett
                if cpoints_dbg:
                    A_ = cpoints_dbg["A"]
                    B_ = cpoints_dbg["B"]
                    C_ = cpoints_dbg["C"]
                    D_calc_= calculate_point_D(A_, B_, C_)
                    D_corr_= adjust_point_D(A_, B_, C_, D_calc_, PERCENT_AB, PERCENT_BC)
                    sorted_pts_dbg= sort_points(A_, B_, C_, D_corr_.astype(int))
                    warped_dbg, M_dbg= warp_perspective(first_frame, sorted_pts_dbg)

                    step_= 800 // 8
                    for i_ in range(9):
                        cv2.line(warped_dbg, (0, i_*step_), (800, i_*step_), (0,0,0), 2)
                        cv2.line(warped_dbg, (i_*step_, 0), (i_*step_, 800), (0,0,0), 2)

                    if midpoints.shape[0] > 0:
                        ones_dbg= np.ones((midpoints.shape[0],1))
                        mh_dbg= np.hstack([midpoints, ones_dbg])
                        t_dbg= M_dbg @ mh_dbg.T
                        t_dbg/= t_dbg[2,:]
                        t_dbg= t_dbg[:2,:].T
                        for (mx_, my_), lbl_ in zip(t_dbg, labels):
                            fen_c= FEN_MAPPING.get(lbl_, '?')
                            px_, py_= int(mx_), int(my_)
                            cv2.circle(warped_dbg, (px_, py_), 4, (255, 0, 0), -1)
                            cv2.putText(warped_dbg, fen_c,
                                        (px_+5, py_+5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0,
                                        (255, 0, 0),
                                        2)

                    st.subheader("Entzerrtes Schachbrett mit Figurenpositionen")
                    st.image(warped_dbg, channels="BGR")

        # =========================
        # Suche Spielstart
        # =========================
        st.write("Suche Spielstart...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        found_start= False
        white_side= None

        while True:
            ret, ft= cap.read()
            if not ret:
                break
            midp__, labs__, conf__, _= detect_pieces(ft, piece_model, PIECE_DETECTION_CONFIG)
            if midp__.shape[0]==0:
                continue
            ones__= np.ones((midp__.shape[0],1))
            mh__= np.hstack([midp__, ones__])
            tf__= M_ @ mh__.T
            tf__/= tf__[2,:]
            tf__= tf__[:2,:].T

            for side_ in ["links","rechts"]:
                if side_=="rechts":
                    ro__= np.zeros_like(tf__)
                    ro__[:,0]= 800 - tf__[:,0]
                    ro__[:,1]= 800 - tf__[:,1]
                else:
                    ro__= tf__

                fen__= generate_placement_from_board(ro__, labs__)
                if fen__ == STARTING_PLACEMENT:
                    white_side= side_
                    found_start= True
                    st.success(f"Spielstart erkannt - Weiss spielt {white_side}")
                    break

            if found_start:
                break

        if not found_start:
            st.warning("Keine automatische Grundstellung. Manuell wählen:")
            user_side= st.radio("Wo ist Weiss?", ["links","rechts"])
            white_side= user_side
            st.info(f"Setze Weiss auf {white_side}.")

        st.write("Analysiere Spielverlauf... Dies kann einige Minuten dauern...")

        # =========================
        # Haupt-Schleife: Zustände
        # =========================

        state_info= {
            'state': "normal",
            'frame_number': 0,
            'previous_fen': STARTING_PLACEMENT if found_start else global_board.fen().split(" ")[0],
            'color': 'w',
            'global_board': global_board,
            'M': M_,
            'white_side': white_side,
            'roi': None,
            'detected_clock_changes': [],
            'move_label': None,
            'halfmove_count': 1,
            'frame_X': None,
            'frame_Z': None,
            'frames_checked_total': 0,
            'previous_player_turn': None,
            'stockfish_available': stockfish_available,
            'search_frame_count': 0,
            'fix_frame_count': 0,
            'game': game,
            'node': node
        }

        # --- Deine process_* -Funktionen (normal/searching/fixing),
        # --- in denen wir nun "Gespielter Zug" + "Empfohlener nächster Zug" an den passenden Stellen eingefügt haben.

        def process_normal_state(frame, frame_number, cap, params, models, stinfo):
            piece_model, corner_model, clock_model= models
            board= stinfo['global_board']
            color= stinfo['color']
            prev_fen= stinfo['previous_fen']
            M_= stinfo['M']
            white_side= stinfo['white_side']
            node= stinfo['node']
            roi= stinfo['roi']

            if roi:
                cinfo_all= detect_clock(frame, clock_model, CLOCK_DETECTION_ROI_CONFIG)
                cinfo= []
                for ci in cinfo_all:
                    x1c,y1c,x2c,y2c= ci['bbox']
                    if (roi[0]<= x1c<= roi[2]) and (roi[1]<= y1c<= roi[3]):
                        cinfo.append(ci)
            else:
                cinfo= detect_clock(frame, clock_model, CLOCK_DETECTION_CONFIG)

            pturn= None
            for ci in cinfo:
                if ci['label'] in ["left","right"]:
                    pturn= ci['label']
                    break
                elif ci['label']=="hold":
                    pturn="hold"
                    break

            # Wechsel
            if pturn and pturn!="hold" and pturn!= stinfo['previous_player_turn']:
                stinfo['detected_clock_changes'].append(frame_number)
                stinfo['frame_X']= frame_number
                stinfo['previous_player_turn']= pturn

                midpX, labsX, confsX, _= detect_pieces(frame, piece_model, PIECE_DETECTION_CONFIG)
                if midpX.shape[0]==0:
                    st.warning("Keine Figuren erkannt -> wechsle in searching")
                    stinfo['state']="searching"
                    jump_= params['frame_interval_normal']
                    newpos_= max(frame_number-jump_, 0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, newpos_)
                    stinfo['frame_number']= newpos_
                    return stinfo

                onesX= np.ones((midpX.shape[0],1))
                mhX= np.hstack([midpX, onesX])
                transfX= M_ @ mhX.T
                transfX/= transfX[2,:]
                transfX= transfX[:2,:].T

                if white_side=="rechts":
                    rotX= np.zeros_like(transfX)
                    rotX[:,0]= 800- transfX[:,0]
                    rotX[:,1]= 800- transfX[:,1]
                else:
                    rotX= transfX

                cfenX= generate_placement_from_board(rotX, labsX)
                if cfenX== prev_fen:
                    st.info("FEN unverändert -> bleibe normal")
                else:
                    mvX= fen_diff_to_move(prev_fen, cfenX, color=color)
                    if mvX and mvX in board.legal_moves:
                        board.push(mvX)
                        node= node.add_variation(mvX)
                        stinfo['node']= node

                        halfm= stinfo['halfmove_count']
                        # Exakte Zug-Beschriftung wie im alten Code + Überschrift
                        if color=='w':
                            move_label= f"Zug {(halfm+1)//2}a: {mvX}"
                        else:
                            move_label= f"Zug {(halfm+1)//2}b: {mvX}"

                        with st.expander(move_label):
                            st.write(f"**FEN**: {cfenX}")
                            st.markdown("#### Gespielter Zug")  # <--- Gewünschte Überschrift
                            colL, colR= st.columns(2)
                            with colL:
                                st.image(frame, caption=f"Frame {frame_number}", channels="BGR")
                            with colR:
                                fen_for_svg= f"{cfenX} {('w' if color=='b' else 'b')} KQkq - 0 1"
                                bsvg= create_chessboard_svg(fen_for_svg, last_move= mvX)
                                st.components.v1.html(bsvg, height=400)

                                if stinfo['stockfish_available']:
                                    keyA= f"analysis_{fen_for_svg}_{halfm}"
                                    if keyA not in st.session_state:
                                        st.session_state[keyA]= analyze_fen_with_stockfish(fen_for_svg)
                                    bestM, evaluation, mate= st.session_state[keyA]
                                    if mate and mate!=0:
                                        st.info(f"**Matt in {mate} Zügen**!")
                                    elif evaluation!="None":
                                        try:
                                            ev_val= float(evaluation)
                                            ev_cl= max(-10,min(10,ev_val))
                                            pr_= (ev_cl+10)/20
                                            bar_ = f"""
                                            <div style='width:300px;height:50px;position:relative;border:2px solid #333;
                                            background:linear-gradient(to right,white {pr_*100}%,black {pr_*100}% 100%);
                                            margin-top:10px;'>
                                              <div style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                                              color:#aaa;font-weight:bold;font-size:1.1em;'>{ev_val:.2f}</div>
                                            </div>
                                            """
                                            st.markdown(bar_, unsafe_allow_html=True)
                                        except:
                                            pass
                                    if bestM:
                                        st.markdown("#### Empfohlener nächster Zug")
                                        st.write(f"**Empfohlener nächster Zug**: {bestM}")
                                        bsvg2= create_chessboard_svg_with_bestmove(fen_for_svg,bestM)
                                        st.components.v1.html(bsvg2, height=400)

                        if board.is_game_over():
                            stinfo['state']="finished"
                            return stinfo

                        stinfo['color']= 'b' if color=='w' else 'w'
                        stinfo['previous_fen']= cfenX
                        stinfo['halfmove_count']+=1

                        if roi is None and len(cinfo)>0:
                            box_= cinfo[0]['bbox']
                            fh,fw= frame.shape[:2]
                            roi_= expand_bbox(box_[0],box_[1],box_[2],box_[3],PADDING, fw,fh)
                            stinfo['roi']= roi_

                    else:
                        st.warning("Zug nicht legal -> wechsle in searching")
                        stinfo['state']="searching"
                        jump__= params['frame_interval_normal']
                        newp__= max(frame_number-jump__, 0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, newp__)
                        stinfo['frame_number']= newp__

            return stinfo

        def process_searching_state(frame, frame_number, cap, params, models, stinfo):
            piece_model, corner_model, clock_model= models
            board= stinfo['global_board']
            color= stinfo['color']
            prev_fen= stinfo['previous_fen']
            M_= stinfo['M']
            w_side= stinfo['white_side']
            node= stinfo['node']
            roi= stinfo['roi']
            frame_X= stinfo['frame_X']

            if frame_number>= frame_X:
                st.info("Keine exakte Zwischen-Frame gefunden -> normal")
                stinfo['state']="normal"
                return stinfo

            if frame_number % params['frame_interval_search']!=0:
                return stinfo

            if roi:
                cinfo= detect_clock(frame, clock_model, CLOCK_DETECTION_ROI_CONFIG)
                cinfo= [
                    ci for ci in cinfo
                    if (roi[0]<= ci['bbox'][0]<= roi[2] and roi[1]<= ci['bbox'][1]<= roi[3])
                ]
            else:
                cinfo= detect_clock(frame, clock_model, CLOCK_DETECTION_CONFIG)

            pturn_search= None
            for ci in cinfo:
                if ci['label'] in ["left","right"]:
                    pturn_search= ci['label']
                    break
                elif ci['label']=="hold":
                    pturn_search= "hold"
                    break

            if pturn_search== stinfo['previous_player_turn']:
                st.info(f"Exakter Uhrwechsel-Frame {frame_number} gefunden.")
                midZ, lblZ, confZ, _= detect_pieces(frame, piece_model, PIECE_DETECTION_CONFIG)
                if midZ.shape[0]==0:
                    st.warning("Keine Figuren -> normal")
                    stinfo['state']="normal"
                    return stinfo

                onesZ= np.ones((midZ.shape[0],1))
                mhZ= np.hstack([midZ, onesZ])
                transfZ= M_ @ mhZ.T
                transfZ/= transfZ[2,:]
                transfZ= transfZ[:2,:].T

                if w_side=="rechts":
                    rotZ= np.zeros_like(transfZ)
                    rotZ[:,0]= 800- transfZ[:,0]
                    rotZ[:,1]= 800- transfZ[:,1]
                else:
                    rotZ= transfZ

                current_fenZ= generate_placement_from_board(rotZ, lblZ)
                if current_fenZ== prev_fen:
                    st.info("FEN identisch -> normal")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)
                    stinfo['frame_number']= frame_X
                    stinfo['state']="normal"
                    return stinfo
                else:
                    mvZ= fen_diff_to_move(prev_fen, current_fenZ, color=color)
                    if mvZ and mvZ in board.legal_moves:
                        board.push(mvZ)
                        node= node.add_variation(mvZ)
                        stinfo['node']= node

                        halfm= stinfo['halfmove_count']
                        if color=='w':
                            move_label= f"Zug {(halfm+1)//2}a: {mvZ}"
                        else:
                            move_label= f"Zug {(halfm+1)//2}b: {mvZ}"

                        with st.expander(move_label):
                            st.write(f"**FEN**: {current_fenZ}")
                            st.markdown("#### Gespielter Zug")
                            colL, colR= st.columns(2)
                            with colL:
                                st.image(frame, caption=f"Frame {frame_number}", channels="BGR")
                            with colR:
                                fen_for_b= f"{current_fenZ} {('w' if color=='b' else 'b')} KQkq - 0 1"
                                bsvg_= create_chessboard_svg(fen_for_b, last_move= mvZ)
                                st.components.v1.html(bsvg_, height=400)

                                if stinfo['stockfish_available']:
                                    keyS= f"analysis_{fen_for_b}_{halfm}"
                                    if keyS not in st.session_state:
                                        st.session_state[keyS]= analyze_fen_with_stockfish(fen_for_b)
                                    bestS,evS,mtS= st.session_state[keyS]
                                    if mtS and mtS!=0:
                                        st.info(f"Matt in {mtS} Zügen!")
                                    elif evS!="None":
                                        try:
                                            e__= float(evS)
                                            ecl__= max(-10,min(10,e__))
                                            pr__= (ecl__+10)/20
                                            barS= f"""
                                            <div style='width:300px;height:50px;position:relative;border:2px solid #333;
                                            background:linear-gradient(to right,white {pr__*100}%,black {pr__*100}% 100%);
                                            margin-top:10px;'>
                                                  <div style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                                                  color:#aaa;font-weight:bold;font-size:1.1em;'>{e__:.2f}</div>
                                            </div>
                                            """
                                            st.markdown(barS, unsafe_allow_html=True)
                                        except:
                                            pass
                                    if bestS:
                                        st.markdown("#### Empfohlener nächster Zug")
                                        st.write(f"**Empfohlener nächster Zug**: {bestS}")
                                        bsvS2= create_chessboard_svg_with_bestmove(fen_for_b,bestS)
                                        st.components.v1.html(bsvS2, height=400)

                        if board.is_game_over():
                            stinfo['state']="finished"
                            return stinfo

                        stinfo['color']= 'b' if color=='w' else 'w'
                        stinfo['previous_fen']= current_fenZ
                        stinfo['halfmove_count']+=1

                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)
                        stinfo['frame_number']= frame_X
                        stinfo['state']="normal"
                        return stinfo
                    else:
                        st.warning("Nicht legal -> wechsle in fixing")
                        stinfo['state']="fixing"
                        stinfo['frame_Z']= frame_number
                        stinfo['frames_checked_total']= 0
                        return stinfo

            return stinfo

        def process_fixing_state(frame, frame_number, cap, params, models, stinfo):
            piece_model, corner_model, clock_model= models
            board= stinfo['global_board']
            color= stinfo['color']
            prev_fen= stinfo['previous_fen']
            M_= stinfo['M']
            w_side= stinfo['white_side']
            frame_X= stinfo['frame_X']
            frames_checked_total= stinfo['frames_checked_total']
            node= stinfo['node']

            if frames_checked_total> params['max_search_frames']:
                st.warning("Maximale Versuche im Fix-Zustand -> zurück zu normal")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)
                stinfo['frame_number']= frame_X
                stinfo['state']="normal"
                return stinfo

            if frame_number % params['frame_interval_fix']!=0:
                return stinfo

            frames_checked_total+=1
            stinfo['frames_checked_total']= frames_checked_total

            midFix, lblFix, confFix, _= detect_pieces(frame, piece_model, PIECE_DETECTION_CONFIG)
            if midFix.shape[0]==0:
                return stinfo

            onesF= np.ones((midFix.shape[0],1))
            mhF= np.hstack([midFix, onesF])
            transfF= M_ @ mhF.T
            transfF/= transfF[2,:]
            transfF= transfF[:2,:].T

            if w_side=="rechts":
                rotF= np.zeros_like(transfF)
                rotF[:,0]=800- transfF[:,0]
                rotF[:,1]=800- transfF[:,1]
            else:
                rotF= transfF

            fenFix= generate_placement_from_board(rotF, lblFix)
            if fenFix== prev_fen:
                st.info("FEN unverändert -> normal")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)
                stinfo['frame_number']= frame_X
                stinfo['state']="normal"
                return stinfo

            mvFix= fen_diff_to_move(prev_fen, fenFix, color=color)
            if mvFix and mvFix in board.legal_moves:
                board.push(mvFix)
                node= node.add_variation(mvFix)
                stinfo['node']= node

                halfm= stinfo['halfmove_count']
                if color=='w':
                    move_label= f"Zug {(halfm+1)//2}a: {mvFix}"
                else:
                    move_label= f"Zug {(halfm+1)//2}b: {mvFix}"

                with st.expander(move_label):
                    st.write(f"**FEN**: {fenFix}")
                    st.markdown("#### Gespielter Zug")
                    colL, colR= st.columns(2)
                    with colL:
                        st.image(frame, caption=f"Frame {frame_number}", channels="BGR")
                    with colR:
                        fen_for_svg= f"{fenFix} {('w' if color=='b' else 'b')} KQkq - 0 1"
                        bsvg_= create_chessboard_svg(fen_for_svg, last_move= mvFix)
                        st.components.v1.html(bsvg_, height=400)

                        if stinfo['stockfish_available']:
                            keyF= f"analysis_{fen_for_svg}_{halfm}"
                            if keyF not in st.session_state:
                                st.session_state[keyF]= analyze_fen_with_stockfish(fen_for_svg)
                            bestF, evF, mtF= st.session_state[keyF]
                            if mtF and mtF!=0:
                                st.info(f"Matt in {mtF} Zügen!")
                            elif evF!="None":
                                try:
                                    valFF= float(evF)
                                    valFF_cl= max(-10,min(10,valFF))
                                    prFF= (valFF_cl+10)/20
                                    barFF= f"""
                                    <div style='width:300px;height:50px;position:relative;border:2px solid #333;
                                    background:linear-gradient(to right,white {prFF*100}%,black {prFF*100}% 100%);
                                    margin-top:10px;'>
                                      <div style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                                      color:#aaa;font-weight:bold;font-size:1.1em;'>{valFF:.2f}</div>
                                    </div>
                                    """
                                    st.markdown(barFF, unsafe_allow_html=True)
                                except:
                                    pass
                            if bestF:
                                st.markdown("#### Empfohlener nächster Zug")
                                st.write(f"**Empfohlener nächster Zug**: {bestF}")
                                bsvgBestF= create_chessboard_svg_with_bestmove(fen_for_svg,bestF)
                                st.components.v1.html(bsvgBestF, height=400)

                if board.is_game_over():
                    stinfo['state']="finished"
                    return stinfo

                stinfo['color']= 'b' if color=='w' else 'w'
                stinfo['previous_fen']= fenFix
                stinfo['halfmove_count']+=1

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)
                stinfo['frame_number']= frame_X
                stinfo['state']="normal"

            return stinfo

        # ============ Hauptloop ============
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            if state_info['state']=="finished":
                break

            ret, frame= cap.read()
            if not ret:
                st.info("Video zu Ende, keine weiteren Züge erkannt.")
                break

            state_info['frame_number']+=1
            fn= state_info['frame_number']
            ratio= fn/(total_frames if total_frames>0 else 1)

            # In der Sidebar statt im Hauptbereich
            progress_bar.progress(min(ratio,1.0))
            frame_counter_placeholder.text(f"Analysierter Frame: {fn} / {total_frames}  ({ratio*100:.1f} %)")

            if state_info['state']=="normal":
                if fn % frame_interval_normal==0:
                    state_info= process_normal_state(
                        frame, fn, cap,
                        {
                            'frame_interval_normal': frame_interval_normal,
                            'frame_interval_search': frame_interval_search,
                            'frame_interval_fix': frame_interval_fix,
                            'max_search_frames': MAX_FRAMES_FIX
                        },
                        (piece_model, corner_model, clock_model),
                        state_info
                    )

            elif state_info['state']=="searching":
                state_info= process_searching_state(
                    frame, fn, cap,
                    {
                        'frame_interval_normal': frame_interval_normal,
                        'frame_interval_search': frame_interval_search,
                        'frame_interval_fix': frame_interval_fix,
                        'max_search_frames': MAX_FRAMES_FIX
                    },
                    (piece_model, corner_model, clock_model),
                    state_info
                )
            elif state_info['state']=="fixing":
                state_info= process_fixing_state(
                    frame, fn, cap,
                    {
                        'frame_interval_normal': frame_interval_normal,
                        'frame_interval_search': frame_interval_search,
                        'frame_interval_fix': frame_interval_fix,
                        'max_search_frames': MAX_FRAMES_FIX
                    },
                    (piece_model, corner_model, clock_model),
                    state_info
                )

            if state_info['state']=="finished":
                break

            if state_info['global_board'].is_game_over():
                state_info['state']="finished"
                break

        cap.release()

        # =========================
        # (Optionale) Rückwärtssuche
        # =========================
        if not state_info['global_board'].is_game_over():
            st.info("Video zu Ende, kein Matt erkannt. Starte Rückwärtssuche...")
            cap_back= cv2.VideoCapture(temp_video_path)
            state_info= search_backwards_for_mate_or_matt(
                cap_back,
                start_frame= state_info['frame_number'],
                max_frames_fix= MAX_FRAMES_FIX,
                piece_model= piece_model,
                state_info= state_info,
                frame_interval_search= frame_interval_search
            )
            cap_back.release()

        # =========================
        # Finale Ausgabe (nun auf 100% setzen)
        # =========================
        progress_bar.progress(1.0)  # <-- auf 100% springen, wenn Analyse abgeschlossen
        frame_counter_placeholder.text(f"Analysierter Frame: {total_frames} / {total_frames} (100.0 %)")

        if state_info['global_board'].is_game_over():
            if state_info['global_board'].is_checkmate():
                winner= "Weiss" if state_info['color']=="b" else "Schwarz"
                st.success(f"Glückwunsch, {winner} hat gewonnen!")
            else:
                st.info("Partie ist zu Ende (z.B. Patt oder Abbruch).")
        else:
            st.info("Keine Matt-Situation gefunden, auch rückwärts nicht.")

        # PGN ausgeben
        raw_pgn= str(state_info['game'])
        moves_only_pgn= remove_pgn_headers(raw_pgn)

        st.markdown(
            """<h2 style='text-align:left; font-size:1.6em; color:#b00; margin-top:40px;'>
               Ermittelte PGN
               </h2>""",
            unsafe_allow_html=True
        )
        st.code(moves_only_pgn, language="plaintext")
        st.balloons()

        try:
            os.remove(temp_video_path)
        except:
            pass

import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import os
import chess
import chess.pgn
import chess.svg
import logging
import tempfile
import requests
import time

# ==============================
# Logging Konfiguration
# ==============================
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ==============================
# EXAKTE UI: Wie im alten Code
# + deine neuen Wünsche
# ==============================
st.set_page_config(page_title="CaroCam - Schach-Tracker", layout="wide")
st.image("CaroCam Logo.png", use_container_width=False, width=450)

st.title("KI-gestützte Schach-Analyse")
st.markdown("**Eine Projektarbeit von Luca Lenherr und Dennis Langer**")

# ==============================
# Konfigurationsparameter
# ==============================
PERCENT_AB = 0.17
PERCENT_BC = -0.07

INTERVAL_NORMAL_SECONDS = 1.0   # Zustand 1 (Normal)
INTERVAL_SEARCH_SECONDS = 0.2   # Zustand 2 (Searching)
INTERVAL_FIX_SECONDS = 0.15     # Zustand 3 (Fixing)
MAX_FRAMES_FIX = 80            # maximale Frames im Fixierzustand
MIN_CONF_PIECE = 0.7           # Mindestkonfidenz
PADDING = 30                   # Padding für ROI

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

# Figuren-Labels mit Farben (für Bounding Boxes)
PIECE_COLORS = {
    'White Pawn':   (255, 255, 0),
    'White Knight': (255, 153, 51),
    'White Bishop': (0, 255, 255),
    'White Rook':   (0, 255, 0),
    'White Queen':  (255, 0, 255),
    'White King':   (255, 128, 128),
    'Black Pawn':   (128, 128, 128),
    'Black Knight': (0, 0, 255),
    'Black Bishop': (102, 0, 204),
    'Black Rook':   (255, 0, 0),
    'Black Queen':  (255, 255, 255),
    'Black King':   (128, 0, 128)
}

# YOLO-Configs
PIECE_DETECTION_CONFIG = {
    'min_conf': MIN_CONF_PIECE,
    'conf': 0.05,
    'iou': 0.3,
    'imgsz': 1400,
    'verbose': False
}
CORNER_DETECTION_CONFIG = {
    'conf': 0.1,
    'iou': 0.3,
    'imgsz': 1400,
    'verbose': False
}
CLOCK_DETECTION_CONFIG = {
    'conf': 0.5,
    'iou': 0.6,
    'imgsz': 1200,
    'verbose': False
}
CLOCK_DETECTION_ROI_CONFIG = {
    'conf': 0.5,
    'iou': 0.7,
    'imgsz': 700,
    'verbose': False
}

CLOCK_LABEL_SYNONYMS = {
    'links': 'left',
    'rechts': 'right',
    'leftclock': 'left',
    'rightclock': 'right'
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
        return None, "None", None
    except:
        return None, "None", None

# ==============================
# SVG-Erzeugung
# ==============================
import chess.svg

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
        ksq = board.king(board.turn)
        if ksq is not None:
            check_square = ksq

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
# YOLO-Modelle
# ==============================
@st.cache_resource(show_spinner=True)
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    piece_model_path = os.path.join(
        base_dir, 'Figure Detection Model', 'runs', 'detect', 'yolov8-chess32', 'weights', 'best.pt'
    )
    corner_model_path = os.path.join(
        base_dir, 'Corner Detection Model', 'runs', 'detect', 'yolov8n_corner14', 'weights', 'best.pt'
    )
    clock_model_path = os.path.join(
        base_dir, 'Clock Detection Model', 'runs', 'detect', 'yolov8-chess6', 'weights', 'best.pt'
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
def detect_pieces(image, piece_model, config):
    results = piece_model.predict(
        image,
        conf=config['conf'],
        iou=config['iou'],
        imgsz=config['imgsz'],
        verbose=config['verbose']
    )
    class_names = piece_model.model.names
    res0 = results[0]

    midpoints, labels, confidences = [], [], []
    for box in res0.boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        c_id= int(box.cls.cpu().numpy()[0])
        conf_val= float(box.conf.cpu().numpy()[0])
        label = class_names[c_id]

        if conf_val < config['min_conf']:
            continue

        mid_x = x1 + (x2 - x1) / 2
        mid_y = y1 + (y2 - y1) * 0.75  # kleiner Offset nach unten

        midpoints.append([mid_x, mid_y])
        labels.append(label)
        confidences.append(conf_val)
    return np.array(midpoints), labels, confidences, res0

def detect_corners(image, corner_model, config):
    results = corner_model.predict(
        image,
        conf=config['conf'],
        iou=config['iou'],
        imgsz=config['imgsz'],
        verbose=config['verbose']
    )
    points = {}
    for box in results[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        c_id= int(box.cls.cpu().numpy()[0])
        if c_id == 0:
            points["A"] = np.array([cx,cy])
        elif c_id == 1:
            points["B"] = np.array([cx,cy])
        elif c_id == 2:
            points["C"] = np.array([cx,cy])
    if len(points) != 3:
        return None, None
    return points, results[0]

def detect_clock(image, clock_model, config):
    results = clock_model.predict(
        image,
        conf=config['conf'],
        iou=config['iou'],
        imgsz=config['imgsz'],
        verbose=config['verbose']
    )
    c_info = []
    for box in results[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        cid= int(box.cls.cpu().numpy()[0])
        conf_= float(box.conf.cpu().numpy()[0])
        raw_label= clock_model.model.names[cid].lower()
        if raw_label in CLOCK_LABEL_SYNONYMS:
            raw_label = CLOCK_LABEL_SYNONYMS[raw_label]
        c_info.append({
            'bbox':[x1,y1,x2,y2],
            'label': raw_label,
            'confidence': conf_
        })
    return c_info

def calculate_point_D(A,B,C):
    BC = C - B
    return A + BC

def adjust_point_D(A,B,C, D_calc, pAB,pBC):
    AB = B - A
    BC = C - B
    corr = pAB * AB + pBC * BC
    return D_calc + corr

def sort_points(A,B,C,D):
    pts= np.array([A,B,C,D])
    pts= sorted(pts, key=lambda x:x[1])  # nach y sortieren
    top= sorted(pts[:2], key=lambda x:x[0])
    bot= sorted(pts[2:], key=lambda x:x[0])
    A_,D_ = top
    B_,C_ = bot
    return np.array([A_, B_, C_, D_], dtype=np.float32)

def warp_perspective(img, src_points):
    dst_size = 800
    dst_pts= np.array([
        [0,0],
        [0,dst_size-1],
        [dst_size-1,dst_size-1],
        [dst_size-1,0]
    ], dtype=np.float32)
    M= cv2.getPerspectiveTransform(src_points, dst_pts)
    warped= cv2.warpPerspective(img, M, (dst_size,dst_size))
    # Brett wird rotiert, damit A oben links liegt
    warped= cv2.rotate(warped, cv2.ROTATE_180)
    return warped, M

def generate_placement_from_board(transformed, labels, grid_size=8):
    board = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    step_size= 800 // grid_size
    for (mx,my), lbl in zip(transformed, labels):
        c= int(my // step_size)
        r= 7 - int(mx // step_size)
        if 0 <= r < 8 and 0 <= c < 8:
            fen_char= FEN_MAPPING.get(lbl,'')
            if fen_char:
                board[r][c] = fen_char

    fen_rows = []
    for row_ in board:
        empty_count= 0
        fen_row= ''
        for sq in row_:
            if sq=='':
                empty_count += 1
            else:
                if empty_count>0:
                    fen_row += str(empty_count)
                    empty_count= 0
                fen_row+= sq
        if empty_count>0:
            fen_row+= str(empty_count)
        fen_rows.append(fen_row)
    return '/'.join(fen_rows)

def fen_diff_to_move(fen1, fen2, color='w'):
    if not fen1 or not fen2:
        return None
    full_fen1= f"{fen1} {color} KQkq - 0 1"
    try:
        board= chess.Board(full_fen1)
    except:
        return None
    for mv in board.legal_moves:
        board.push(mv)
        fenp= board.fen().split(" ")[0]
        if fenp == fen2:
            return mv
        board.pop()
    return None

def remove_pgn_headers(pgn_string):
    lines= pgn_string.split("\n")
    cleaned=[]
    for line in lines:
        if line.startswith("[") or line.strip()=="":
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def expand_bbox(x1,y1,x2,y2,padding, fw,fh):
    x1n= max(0, int(x1)-padding)
    y1n= max(0, int(y1)-padding)
    x2n= min(int(fw), int(x2)+padding)
    y2n= min(int(fh), int(y2)+padding)
    return (x1n,y1n,x2n,y2n)

# ==============================
# Rückwärts-Suche
# (Einmaliges "Glückwunsch..." entfernt)
# ==============================
def search_backwards_for_mate_or_matt(cap, start_frame, max_frames_fix, piece_model, state_info, frame_interval_search):
    global_board= state_info['global_board']
    color= state_info['color']
    M_= state_info['M']
    w_side= state_info['white_side']
    prev_fen= state_info['previous_fen']
    node= state_info['node']
    halfmove_count= state_info['halfmove_count']

    current_frame= start_frame
    count_attempts= 0

    while current_frame>=0:
        if count_attempts> max_frames_fix:
            st.info("Rückwärtssuche abgebrochen: max_frames_fix erreicht.")
            break
        count_attempts+=1

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame_b= cap.read()
        if not ret:
            current_frame-= frame_interval_search
            continue

        # Figuren erkennen
        midb, labsb, confsb, _= detect_pieces(frame_b, piece_model, PIECE_DETECTION_CONFIG)
        if midb.shape[0]==0:
            current_frame-= frame_interval_search
            continue

        onesb= np.ones((midb.shape[0],1))
        mhb= np.hstack([midb, onesb])
        transfb= M_ @ mhb.T
        transfb/= transfb[2,:]
        transfb= transfb[:2,:].T

        if w_side=='rechts':
            rotb= np.zeros_like(transfb)
            rotb[:,0]= 800 - transfb[:,0]
            rotb[:,1]= 800 - transfb[:,1]
        else:
            rotb= transfb

        fenb= generate_placement_from_board(rotb, labsb)
        mvb= fen_diff_to_move(prev_fen, fenb, color=color)
        if mvb and mvb in global_board.legal_moves:
            global_board.push(mvb)
            node= node.add_variation(mvb)

            with st.expander(f"Zug {(halfmove_count+1)//2}{'a' if color=='w' else 'b'}: {mvb}"):
                # Gewünschter Titel
                st.markdown("#### Gespielter Zug")
                colL, colR= st.columns(2)
                with colL:
                    st.image(frame_b, caption=f"Frame {current_frame} (Rückwärtssuche)", channels="BGR")
                with colR:
                    fen_for_board= f"{fenb} {('w' if color=='b' else 'b')} KQkq - 0 1"
                    bsvg= create_chessboard_svg(fen_for_board, last_move= mvb)
                    st.markdown("**Vorheriger Zug**", unsafe_allow_html=True)
                    st.write(f"**FEN**: {fenb}")
                    st.components.v1.html(bsvg, height=400)

                    if state_info['stockfish_available']:
                        keyA= f"analysis_{fen_for_board}_{halfmove_count}"
                        if keyA not in st.session_state:
                            st.session_state[keyA]= analyze_fen_with_stockfish(fen_for_board)
                        bestM,evv,mate= st.session_state[keyA]
                        if mate and mate!=0:
                            st.info(f"**Matt in {mate} Zügen**!")
                        elif evv!="None":
                            try:
                                e_val= float(evv)
                                e_cl= max(-10,min(10,e_val))
                                pr_= (e_cl+10)/20
                                bar_ = f"""
                                <div style='width:300px;height:50px;position:relative;border:2px solid #333;
                                background:linear-gradient(to right,white {pr_*100}%,black {pr_*100}% 100%);
                                margin-top:10px;'>
                                  <div style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                                  color:#aaa;font-weight:bold;font-size:1.1em;'>{e_val:.2f}</div>
                                </div>
                                """
                                st.markdown(bar_, unsafe_allow_html=True)
                            except:
                                pass
                        if bestM:
                            st.markdown("#### Empfohlener nächster Zug")
                            st.write(f"**Empfohlener nächster Zug**: {bestM}")
                            bsvg2= create_chessboard_svg_with_bestmove(fen_for_board,bestM)
                            st.components.v1.html(bsvg2, height=400)

            if global_board.is_game_over():
                state_info['state']="finished"
                return state_info

            state_info['color']= 'b' if color=='w' else 'w'
            state_info['previous_fen']= fenb
            state_info['halfmove_count']+=1

        current_frame-= frame_interval_search

    st.info("Rückwärtssuche beendet: kein Matt gefunden.")
    return state_info


# =====================================
# MAIN-Funktion (Zustandsmaschine etc.)
# =====================================
def main():
    # --- Stockfish-Verfügbarkeit checken
    stockfish_available = check_stockfish_api_available()
    if stockfish_available:
        st.success("Stockfish-API ist verfügbar.")
    else:
        st.warning("Stockfish-API ist NICHT verfügbar.")

    # --- Modelle laden
    piece_model, corner_model, clock_model= load_models()

    # --- Sidebar
    st.sidebar.header("Video-Upload")
    video_file= st.sidebar.file_uploader("Bitte ein Schachvideo hochladen", type=["mp4","mov","avi"])
    start_button= st.sidebar.button("Erkennung starten")

    if not video_file:
        st.info("Bitte lade ein Video hoch, um zu starten.")
        return

    # Sidebar: Fortschritts-Anzeige und Frame-Zähler
    progress_bar = st.sidebar.progress(0)
    frame_counter_placeholder = st.sidebar.empty()

    # --- Neues Game
    game= chess.pgn.Game()
    node= game
    global_board= chess.Board()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        temp_video_path = tmp.name
        tmp.write(video_file.read())

    if start_button:
        cap= cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Konnte das Video nicht öffnen.")
            return

        fps= cap.get(cv2.CAP_PROP_FPS)
        if fps<=0:
            fps=25.0

        frame_interval_normal= max(1, int(fps*INTERVAL_NORMAL_SECONDS))
        frame_interval_search= max(1, int(fps*INTERVAL_SEARCH_SECONDS))
        frame_interval_fix= max(1, int(fps*INTERVAL_FIX_SECONDS))
        total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # =========================
        # Brett-Ecken detektieren
        # =========================
        st.write("Starte Analyse...")
        st.write("Suche Brett-Ecken...")
        found_corners= False
        M_= None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame_test= cap.read()
            if not ret:
                break
            cpoints,_= detect_corners(frame_test, corner_model, CORNER_DETECTION_CONFIG)
            if cpoints:
                found_corners= True
                A= cpoints["A"]
                B= cpoints["B"]
                C= cpoints["C"]
                D_calc= calculate_point_D(A,B,C)
                D_corr= adjust_point_D(A,B,C, D_calc, PERCENT_AB, PERCENT_BC)
                sorted_pts= sort_points(A,B,C,D_corr.astype(int))
                M_= cv2.getPerspectiveTransform(
                    sorted_pts,
                    np.array([[0,0],[0,799],[799,799],[799,0]], dtype=np.float32)
                )
                st.success("Ecken erkannt.")
                break

        if not found_corners or M_ is None:
            st.error("Keine Brett-Ecken erkannt. Abbruch.")
            cap.release()
            os.remove(temp_video_path)
            return

        # =========================
        # Expander: Hinter den Kulissen
        # =========================
        with st.expander("Hinter den Kulissen"):
            st.write("Hier siehst du, wie das Schachbrett analysiert wird.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame= cap.read()
            if not ret:
                st.warning("Konnte keinen Frame lesen.")
            else:
                cpoints_dbg, corner_res = detect_corners(first_frame, corner_model, CORNER_DETECTION_CONFIG)
                midpoints, labels, confs, piece_res = detect_pieces(first_frame, piece_model, PIECE_DETECTION_CONFIG)
                cinfo_= detect_clock(first_frame, clock_model, CLOCK_DETECTION_CONFIG)

                preview= first_frame.copy()

                # 1) Eckpunkte + Bounding Box
                if corner_res and len(corner_res.boxes) > 0:
                    for box_ in corner_res.boxes:
                        x1, y1, x2, y2 = box_.xyxy[0].cpu().numpy().astype(int)
                        cx_ = (x1 + x2)//2
                        cy_ = (y1 + y2)//2
                        cid_ = int(box_.cls.cpu().numpy()[0])
                        corner_label = corner_model.model.names[cid_]

                        # Bounding Box (Pink)
                        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        # Mittelpunkt (andere Farbe, z.B. Gelb)
                        cv2.circle(preview, (cx_, cy_), 6, (0, 255, 255), -1)
                        # Text
                        cv2.putText(
                            preview, f"{corner_label} (Mittelpunkt)",
                            (x1, max(0, y2 + 20)),   # Unterhalb des BBox
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 255),
                            2
                        )

                # 2) Korrekturvektor: D_calc, D_corr
                if cpoints_dbg:
                    A_ = cpoints_dbg["A"]
                    B_ = cpoints_dbg["B"]
                    C_ = cpoints_dbg["C"]
                    D_calc_ = calculate_point_D(A_, B_, C_)
                    D_corr_ = adjust_point_D(A_, B_, C_, D_calc_, PERCENT_AB, PERCENT_BC)

                    # Pfeil von D_calc_ zu D_corr_
                    cv2.arrowedLine(
                        preview,
                        (int(D_calc_[0]), int(D_calc_[1])),
                        (int(D_corr_[0]), int(D_corr_[1])),
                        (0, 255, 0), 3
                    )
                    # Kreise + Labels
                    cv2.circle(preview, (int(D_calc_[0]), int(D_calc_[1])), 10, (0, 255, 255), -1)
                    cv2.putText(
                        preview, "D_calc",
                        (int(D_calc_[0] + 10), int(D_calc_[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )
                    cv2.circle(preview, (int(D_corr_[0]), int(D_corr_[1])), 10, (0, 0, 255), -1)
                    cv2.putText(
                        preview, "D_corr",
                        (int(D_corr_[0] + 10), int(D_corr_[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                # 3) Figuren-BoundingBoxes
                if piece_res and len(piece_res.boxes) > 0:
                    for box_ in piece_res.boxes:
                        x1, y1, x2, y2 = box_.xyxy[0].cpu().numpy().astype(int)
                        cid_ = int(box_.cls.cpu().numpy()[0])
                        p_label_ = piece_model.model.names[cid_]
                        confv_ = float(box_.conf.cpu().numpy()[0])
                        color_ = PIECE_COLORS.get(p_label_, (255, 0, 0))
                        cv2.rectangle(preview, (x1, y1), (x2, y2), color_, 2)
                        txt_ = f"{p_label_} {confv_*100:.1f}%"
                        cv2.putText(preview, txt_,
                                    (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    color_,
                                    2)

                # 4) Uhr => nur ein Label + ROI-Beschriftung unterhalb der Box
                if cinfo_:
                    best_clock = max(cinfo_, key=lambda c: c['confidence'])
                    x1c, y1c, x2c, y2c = map(int, best_clock['bbox'])
                    cv2.rectangle(preview, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
                    cv2.putText(preview, "clock",
                                (x1c, max(0, y2c + 20)),  # unterhalb der Box
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2)

                    # ROI-BoundingBox
                    fh, fw = first_frame.shape[:2]
                    roi_ = expand_bbox(x1c, y1c, x2c, y2c, PADDING, fw, fh)
                    rx1, ry1, rx2, ry2 = roi_
                    cv2.rectangle(preview, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                    cv2.putText(preview, "ROI der Uhr",
                                (rx1, ry2 + 20),  # unterhalb
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2)

                # Ausgabe
                st.subheader("Alle erkannten Objekte (Eckpunkte des Bretts, Figuren und Uhr, inkl. Korrekturvektor und ROI)")
                st.image(preview, channels="BGR")

                # 5) Entzerrtes Brett
                if cpoints_dbg:
                    A_ = cpoints_dbg["A"]
                    B_ = cpoints_dbg["B"]
                    C_ = cpoints_dbg["C"]
                    D_calc_= calculate_point_D(A_, B_, C_)
                    D_corr_= adjust_point_D(A_, B_, C_, D_calc_, PERCENT_AB, PERCENT_BC)
                    sorted_pts_dbg= sort_points(A_, B_, C_, D_corr_.astype(int))
                    warped_dbg, M_dbg= warp_perspective(first_frame, sorted_pts_dbg)

                    step_= 800 // 8
                    for i_ in range(9):
                        cv2.line(warped_dbg, (0, i_*step_), (800, i_*step_), (0,0,0), 2)
                        cv2.line(warped_dbg, (i_*step_, 0), (i_*step_, 800), (0,0,0), 2)

                    if midpoints.shape[0] > 0:
                        ones_dbg= np.ones((midpoints.shape[0],1))
                        mh_dbg= np.hstack([midpoints, ones_dbg])
                        t_dbg= M_dbg @ mh_dbg.T
                        t_dbg/= t_dbg[2,:]
                        t_dbg= t_dbg[:2,:].T
                        for (mx_, my_), lbl_ in zip(t_dbg, labels):
                            fen_c= FEN_MAPPING.get(lbl_, '?')
                            px_, py_= int(mx_), int(my_)
                            cv2.circle(warped_dbg, (px_, py_), 4, (255, 0, 0), -1)
                            cv2.putText(warped_dbg, fen_c,
                                        (px_+5, py_+5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0,
                                        (255, 0, 0),
                                        2)

                    st.subheader("Entzerrtes Schachbrett mit Figurenpositionen")
                    st.image(warped_dbg, channels="BGR")

        # =========================
        # Suche Spielstart
        # =========================
        st.write("Suche Spielstart...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        found_start= False
        white_side= None

        while True:
            ret, ft= cap.read()
            if not ret:
                break
            midp__, labs__, conf__, _= detect_pieces(ft, piece_model, PIECE_DETECTION_CONFIG)
            if midp__.shape[0]==0:
                continue
            ones__= np.ones((midp__.shape[0],1))
            mh__= np.hstack([midp__, ones__])
            tf__= M_ @ mh__.T
            tf__/= tf__[2,:]
            tf__= tf__[:2,:].T

            for side_ in ["links","rechts"]:
                if side_=="rechts":
                    ro__= np.zeros_like(tf__)
                    ro__[:,0]= 800 - tf__[:,0]
                    ro__[:,1]= 800 - tf__[:,1]
                else:
                    ro__= tf__

                fen__= generate_placement_from_board(ro__, labs__)
                if fen__ == STARTING_PLACEMENT:
                    white_side= side_
                    found_start= True
                    st.success(f"Spielstart erkannt - Weiss spielt {white_side}")
                    break

            if found_start:
                break

        if not found_start:
            st.warning("Keine automatische Grundstellung. Manuell wählen:")
            user_side= st.radio("Wo ist Weiss?", ["links","rechts"])
            white_side= user_side
            st.info(f"Setze Weiss auf {white_side}.")

        st.write("Analysiere Spielverlauf... Dies kann einige Minuten dauern...")

        # =========================
        # Haupt-Schleife: Zustände
        # =========================

        state_info= {
            'state': "normal",
            'frame_number': 0,
            'previous_fen': STARTING_PLACEMENT if found_start else global_board.fen().split(" ")[0],
            'color': 'w',
            'global_board': global_board,
            'M': M_,
            'white_side': white_side,
            'roi': None,
            'detected_clock_changes': [],
            'move_label': None,
            'halfmove_count': 1,
            'frame_X': None,
            'frame_Z': None,
            'frames_checked_total': 0,
            'previous_player_turn': None,
            'stockfish_available': stockfish_available,
            'search_frame_count': 0,
            'fix_frame_count': 0,
            'game': game,
            'node': node
        }

        # --- Deine process_* -Funktionen (normal/searching/fixing),
        # --- in denen wir nun "Gespielter Zug" + "Empfohlener nächster Zug" an den passenden Stellen eingefügt haben.

        def process_normal_state(frame, frame_number, cap, params, models, stinfo):
            piece_model, corner_model, clock_model= models
            board= stinfo['global_board']
            color= stinfo['color']
            prev_fen= stinfo['previous_fen']
            M_= stinfo['M']
            white_side= stinfo['white_side']
            node= stinfo['node']
            roi= stinfo['roi']

            if roi:
                cinfo_all= detect_clock(frame, clock_model, CLOCK_DETECTION_ROI_CONFIG)
                cinfo= []
                for ci in cinfo_all:
                    x1c,y1c,x2c,y2c= ci['bbox']
                    if (roi[0]<= x1c<= roi[2]) and (roi[1]<= y1c<= roi[3]):
                        cinfo.append(ci)
            else:
                cinfo= detect_clock(frame, clock_model, CLOCK_DETECTION_CONFIG)

            pturn= None
            for ci in cinfo:
                if ci['label'] in ["left","right"]:
                    pturn= ci['label']
                    break
                elif ci['label']=="hold":
                    pturn="hold"
                    break

            # Wechsel
            if pturn and pturn!="hold" and pturn!= stinfo['previous_player_turn']:
                stinfo['detected_clock_changes'].append(frame_number)
                stinfo['frame_X']= frame_number
                stinfo['previous_player_turn']= pturn

                midpX, labsX, confsX, _= detect_pieces(frame, piece_model, PIECE_DETECTION_CONFIG)
                if midpX.shape[0]==0:
                    st.warning("Keine Figuren erkannt -> wechsle in searching")
                    stinfo['state']="searching"
                    jump_= params['frame_interval_normal']
                    newpos_= max(frame_number-jump_, 0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, newpos_)
                    stinfo['frame_number']= newpos_
                    return stinfo

                onesX= np.ones((midpX.shape[0],1))
                mhX= np.hstack([midpX, onesX])
                transfX= M_ @ mhX.T
                transfX/= transfX[2,:]
                transfX= transfX[:2,:].T

                if white_side=="rechts":
                    rotX= np.zeros_like(transfX)
                    rotX[:,0]= 800- transfX[:,0]
                    rotX[:,1]= 800- transfX[:,1]
                else:
                    rotX= transfX

                cfenX= generate_placement_from_board(rotX, labsX)
                if cfenX== prev_fen:
                    st.info("FEN unverändert -> bleibe normal")
                else:
                    mvX= fen_diff_to_move(prev_fen, cfenX, color=color)
                    if mvX and mvX in board.legal_moves:
                        board.push(mvX)
                        node= node.add_variation(mvX)
                        stinfo['node']= node

                        halfm= stinfo['halfmove_count']
                        # Exakte Zug-Beschriftung wie im alten Code + Überschrift
                        if color=='w':
                            move_label= f"Zug {(halfm+1)//2}a: {mvX}"
                        else:
                            move_label= f"Zug {(halfm+1)//2}b: {mvX}"

                        with st.expander(move_label):
                            st.write(f"**FEN**: {cfenX}")
                            st.markdown("#### Gespielter Zug")  # <--- Gewünschte Überschrift
                            colL, colR= st.columns(2)
                            with colL:
                                st.image(frame, caption=f"Frame {frame_number}", channels="BGR")
                            with colR:
                                fen_for_svg= f"{cfenX} {('w' if color=='b' else 'b')} KQkq - 0 1"
                                bsvg= create_chessboard_svg(fen_for_svg, last_move= mvX)
                                st.components.v1.html(bsvg, height=400)

                                if stinfo['stockfish_available']:
                                    keyA= f"analysis_{fen_for_svg}_{halfm}"
                                    if keyA not in st.session_state:
                                        st.session_state[keyA]= analyze_fen_with_stockfish(fen_for_svg)
                                    bestM, evaluation, mate= st.session_state[keyA]
                                    if mate and mate!=0:
                                        st.info(f"**Matt in {mate} Zügen**!")
                                    elif evaluation!="None":
                                        try:
                                            ev_val= float(evaluation)
                                            ev_cl= max(-10,min(10,ev_val))
                                            pr_= (ev_cl+10)/20
                                            bar_ = f"""
                                            <div style='width:300px;height:50px;position:relative;border:2px solid #333;
                                            background:linear-gradient(to right,white {pr_*100}%,black {pr_*100}% 100%);
                                            margin-top:10px;'>
                                              <div style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                                              color:#aaa;font-weight:bold;font-size:1.1em;'>{ev_val:.2f}</div>
                                            </div>
                                            """
                                            st.markdown(bar_, unsafe_allow_html=True)
                                        except:
                                            pass
                                    if bestM:
                                        st.markdown("#### Empfohlener nächster Zug")
                                        st.write(f"**Empfohlener nächster Zug**: {bestM}")
                                        bsvg2= create_chessboard_svg_with_bestmove(fen_for_svg,bestM)
                                        st.components.v1.html(bsvg2, height=400)

                        if board.is_game_over():
                            stinfo['state']="finished"
                            return stinfo

                        stinfo['color']= 'b' if color=='w' else 'w'
                        stinfo['previous_fen']= cfenX
                        stinfo['halfmove_count']+=1

                        if roi is None and len(cinfo)>0:
                            box_= cinfo[0]['bbox']
                            fh,fw= frame.shape[:2]
                            roi_= expand_bbox(box_[0],box_[1],box_[2],box_[3],PADDING, fw,fh)
                            stinfo['roi']= roi_

                    else:
                        st.warning("Zug nicht legal -> wechsle in searching")
                        stinfo['state']="searching"
                        jump__= params['frame_interval_normal']
                        newp__= max(frame_number-jump__, 0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, newp__)
                        stinfo['frame_number']= newp__

            return stinfo

        def process_searching_state(frame, frame_number, cap, params, models, stinfo):
            piece_model, corner_model, clock_model= models
            board= stinfo['global_board']
            color= stinfo['color']
            prev_fen= stinfo['previous_fen']
            M_= stinfo['M']
            w_side= stinfo['white_side']
            node= stinfo['node']
            roi= stinfo['roi']
            frame_X= stinfo['frame_X']

            if frame_number>= frame_X:
                st.info("Keine exakte Zwischen-Frame gefunden -> normal")
                stinfo['state']="normal"
                return stinfo

            if frame_number % params['frame_interval_search']!=0:
                return stinfo

            if roi:
                cinfo= detect_clock(frame, clock_model, CLOCK_DETECTION_ROI_CONFIG)
                cinfo= [
                    ci for ci in cinfo
                    if (roi[0]<= ci['bbox'][0]<= roi[2] and roi[1]<= ci['bbox'][1]<= roi[3])
                ]
            else:
                cinfo= detect_clock(frame, clock_model, CLOCK_DETECTION_CONFIG)

            pturn_search= None
            for ci in cinfo:
                if ci['label'] in ["left","right"]:
                    pturn_search= ci['label']
                    break
                elif ci['label']=="hold":
                    pturn_search= "hold"
                    break

            if pturn_search== stinfo['previous_player_turn']:
                st.info(f"Exakter Uhrwechsel-Frame {frame_number} gefunden.")
                midZ, lblZ, confZ, _= detect_pieces(frame, piece_model, PIECE_DETECTION_CONFIG)
                if midZ.shape[0]==0:
                    st.warning("Keine Figuren -> normal")
                    stinfo['state']="normal"
                    return stinfo

                onesZ= np.ones((midZ.shape[0],1))
                mhZ= np.hstack([midZ, onesZ])
                transfZ= M_ @ mhZ.T
                transfZ/= transfZ[2,:]
                transfZ= transfZ[:2,:].T

                if w_side=="rechts":
                    rotZ= np.zeros_like(transfZ)
                    rotZ[:,0]= 800- transfZ[:,0]
                    rotZ[:,1]= 800- transfZ[:,1]
                else:
                    rotZ= transfZ

                current_fenZ= generate_placement_from_board(rotZ, lblZ)
                if current_fenZ== prev_fen:
                    st.info("FEN identisch -> normal")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)
                    stinfo['frame_number']= frame_X
                    stinfo['state']="normal"
                    return stinfo
                else:
                    mvZ= fen_diff_to_move(prev_fen, current_fenZ, color=color)
                    if mvZ and mvZ in board.legal_moves:
                        board.push(mvZ)
                        node= node.add_variation(mvZ)
                        stinfo['node']= node

                        halfm= stinfo['halfmove_count']
                        if color=='w':
                            move_label= f"Zug {(halfm+1)//2}a: {mvZ}"
                        else:
                            move_label= f"Zug {(halfm+1)//2}b: {mvZ}"

                        with st.expander(move_label):
                            st.write(f"**FEN**: {current_fenZ}")
                            st.markdown("#### Gespielter Zug")
                            colL, colR= st.columns(2)
                            with colL:
                                st.image(frame, caption=f"Frame {frame_number}", channels="BGR")
                            with colR:
                                fen_for_b= f"{current_fenZ} {('w' if color=='b' else 'b')} KQkq - 0 1"
                                bsvg_= create_chessboard_svg(fen_for_b, last_move= mvZ)
                                st.components.v1.html(bsvg_, height=400)

                                if stinfo['stockfish_available']:
                                    keyS= f"analysis_{fen_for_b}_{halfm}"
                                    if keyS not in st.session_state:
                                        st.session_state[keyS]= analyze_fen_with_stockfish(fen_for_b)
                                    bestS,evS,mtS= st.session_state[keyS]
                                    if mtS and mtS!=0:
                                        st.info(f"Matt in {mtS} Zügen!")
                                    elif evS!="None":
                                        try:
                                            e__= float(evS)
                                            ecl__= max(-10,min(10,e__))
                                            pr__= (ecl__+10)/20
                                            barS= f"""
                                            <div style='width:300px;height:50px;position:relative;border:2px solid #333;
                                            background:linear-gradient(to right,white {pr__*100}%,black {pr__*100}% 100%);
                                            margin-top:10px;'>
                                                  <div style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                                                  color:#aaa;font-weight:bold;font-size:1.1em;'>{e__:.2f}</div>
                                            </div>
                                            """
                                            st.markdown(barS, unsafe_allow_html=True)
                                        except:
                                            pass
                                    if bestS:
                                        st.markdown("#### Empfohlener nächster Zug")
                                        st.write(f"**Empfohlener nächster Zug**: {bestS}")
                                        bsvS2= create_chessboard_svg_with_bestmove(fen_for_b,bestS)
                                        st.components.v1.html(bsvS2, height=400)

                        if board.is_game_over():
                            stinfo['state']="finished"
                            return stinfo

                        stinfo['color']= 'b' if color=='w' else 'w'
                        stinfo['previous_fen']= current_fenZ
                        stinfo['halfmove_count']+=1

                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)
                        stinfo['frame_number']= frame_X
                        stinfo['state']="normal"
                        return stinfo
                    else:
                        st.warning("Nicht legal -> wechsle in fixing")
                        stinfo['state']="fixing"
                        stinfo['frame_Z']= frame_number
                        stinfo['frames_checked_total']= 0
                        return stinfo

            return stinfo

        def process_fixing_state(frame, frame_number, cap, params, models, stinfo):
            piece_model, corner_model, clock_model= models
            board= stinfo['global_board']
            color= stinfo['color']
            prev_fen= stinfo['previous_fen']
            M_= stinfo['M']
            w_side= stinfo['white_side']
            frame_X= stinfo['frame_X']
            frames_checked_total= stinfo['frames_checked_total']
            node= stinfo['node']

            if frames_checked_total> params['max_search_frames']:
                st.warning("Maximale Versuche im Fix-Zustand -> zurück zu normal")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)
                stinfo['frame_number']= frame_X
                stinfo['state']="normal"
                return stinfo

            if frame_number % params['frame_interval_fix']!=0:
                return stinfo

            frames_checked_total+=1
            stinfo['frames_checked_total']= frames_checked_total

            midFix, lblFix, confFix, _= detect_pieces(frame, piece_model, PIECE_DETECTION_CONFIG)
            if midFix.shape[0]==0:
                return stinfo

            onesF= np.ones((midFix.shape[0],1))
            mhF= np.hstack([midFix, onesF])
            transfF= M_ @ mhF.T
            transfF/= transfF[2,:]
            transfF= transfF[:2,:].T

            if w_side=="rechts":
                rotF= np.zeros_like(transfF)
                rotF[:,0]=800- transfF[:,0]
                rotF[:,1]=800- transfF[:,1]
            else:
                rotF= transfF

            fenFix= generate_placement_from_board(rotF, lblFix)
            if fenFix== prev_fen:
                st.info("FEN unverändert -> normal")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)
                stinfo['frame_number']= frame_X
                stinfo['state']="normal"
                return stinfo

            mvFix= fen_diff_to_move(prev_fen, fenFix, color=color)
            if mvFix and mvFix in board.legal_moves:
                board.push(mvFix)
                node= node.add_variation(mvFix)
                stinfo['node']= node

                halfm= stinfo['halfmove_count']
                if color=='w':
                    move_label= f"Zug {(halfm+1)//2}a: {mvFix}"
                else:
                    move_label= f"Zug {(halfm+1)//2}b: {mvFix}"

                with st.expander(move_label):
                    st.write(f"**FEN**: {fenFix}")
                    st.markdown("#### Gespielter Zug")
                    colL, colR= st.columns(2)
                    with colL:
                        st.image(frame, caption=f"Frame {frame_number}", channels="BGR")
                    with colR:
                        fen_for_svg= f"{fenFix} {('w' if color=='b' else 'b')} KQkq - 0 1"
                        bsvg_= create_chessboard_svg(fen_for_svg, last_move= mvFix)
                        st.components.v1.html(bsvg_, height=400)

                        if stinfo['stockfish_available']:
                            keyF= f"analysis_{fen_for_svg}_{halfm}"
                            if keyF not in st.session_state:
                                st.session_state[keyF]= analyze_fen_with_stockfish(fen_for_svg)
                            bestF, evF, mtF= st.session_state[keyF]
                            if mtF and mtF!=0:
                                st.info(f"Matt in {mtF} Zügen!")
                            elif evF!="None":
                                try:
                                    valFF= float(evF)
                                    valFF_cl= max(-10,min(10,valFF))
                                    prFF= (valFF_cl+10)/20
                                    barFF= f"""
                                    <div style='width:300px;height:50px;position:relative;border:2px solid #333;
                                    background:linear-gradient(to right,white {prFF*100}%,black {prFF*100}% 100%);
                                    margin-top:10px;'>
                                      <div style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                                      color:#aaa;font-weight:bold;font-size:1.1em;'>{valFF:.2f}</div>
                                    </div>
                                    """
                                    st.markdown(barFF, unsafe_allow_html=True)
                                except:
                                    pass
                            if bestF:
                                st.markdown("#### Empfohlener nächster Zug")
                                st.write(f"**Empfohlener nächster Zug**: {bestF}")
                                bsvgBestF= create_chessboard_svg_with_bestmove(fen_for_svg,bestF)
                                st.components.v1.html(bsvgBestF, height=400)

                if board.is_game_over():
                    stinfo['state']="finished"
                    return stinfo

                stinfo['color']= 'b' if color=='w' else 'w'
                stinfo['previous_fen']= fenFix
                stinfo['halfmove_count']+=1

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)
                stinfo['frame_number']= frame_X
                stinfo['state']="normal"

            return stinfo

        # ============ Hauptloop ============
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            if state_info['state']=="finished":
                break

            ret, frame= cap.read()
            if not ret:
                st.info("Video zu Ende, keine weiteren Züge erkannt.")
                break

            state_info['frame_number']+=1
            fn= state_info['frame_number']
            ratio= fn/(total_frames if total_frames>0 else 1)

            # In der Sidebar statt im Hauptbereich
            progress_bar.progress(min(ratio,1.0))
            frame_counter_placeholder.text(f"Analysierter Frame: {fn} / {total_frames}  ({ratio*100:.1f} %)")

            if state_info['state']=="normal":
                if fn % frame_interval_normal==0:
                    state_info= process_normal_state(
                        frame, fn, cap,
                        {
                            'frame_interval_normal': frame_interval_normal,
                            'frame_interval_search': frame_interval_search,
                            'frame_interval_fix': frame_interval_fix,
                            'max_search_frames': MAX_FRAMES_FIX
                        },
                        (piece_model, corner_model, clock_model),
                        state_info
                    )

            elif state_info['state']=="searching":
                state_info= process_searching_state(
                    frame, fn, cap,
                    {
                        'frame_interval_normal': frame_interval_normal,
                        'frame_interval_search': frame_interval_search,
                        'frame_interval_fix': frame_interval_fix,
                        'max_search_frames': MAX_FRAMES_FIX
                    },
                    (piece_model, corner_model, clock_model),
                    state_info
                )
            elif state_info['state']=="fixing":
                state_info= process_fixing_state(
                    frame, fn, cap,
                    {
                        'frame_interval_normal': frame_interval_normal,
                        'frame_interval_search': frame_interval_search,
                        'frame_interval_fix': frame_interval_fix,
                        'max_search_frames': MAX_FRAMES_FIX
                    },
                    (piece_model, corner_model, clock_model),
                    state_info
                )

            if state_info['state']=="finished":
                break

            if state_info['global_board'].is_game_over():
                state_info['state']="finished"
                break

        cap.release()

        # =========================
        # (Optionale) Rückwärtssuche
        # =========================
        if not state_info['global_board'].is_game_over():
            st.info("Video zu Ende, kein Matt erkannt. Starte Rückwärtssuche...")
            cap_back= cv2.VideoCapture(temp_video_path)
            state_info= search_backwards_for_mate_or_matt(
                cap_back,
                start_frame= state_info['frame_number'],
                max_frames_fix= MAX_FRAMES_FIX,
                piece_model= piece_model,
                state_info= state_info,
                frame_interval_search= frame_interval_search
            )
            cap_back.release()

        # =========================
        # Finale Ausgabe (nun auf 100% setzen)
        # =========================
        progress_bar.progress(1.0)  # <-- auf 100% springen, wenn Analyse abgeschlossen
        frame_counter_placeholder.text(f"Analysierter Frame: {total_frames} / {total_frames} (100.0 %)")

        if state_info['global_board'].is_game_over():
            if state_info['global_board'].is_checkmate():
                winner= "Weiss" if state_info['color']=="b" else "Schwarz"
                st.success(f"Glückwunsch, {winner} hat gewonnen!")
            else:
                st.info("Partie ist zu Ende (z.B. Patt oder Abbruch).")
        else:
            st.info("Keine Matt-Situation gefunden, auch rückwärts nicht.")

        # PGN ausgeben
        raw_pgn= str(state_info['game'])
        moves_only_pgn= remove_pgn_headers(raw_pgn)

        st.markdown(
            """<h2 style='text-align:left; font-size:1.6em; color:#b00; margin-top:40px;'>
               Ermittelte PGN
               </h2>""",
            unsafe_allow_html=True
        )
        st.code(moves_only_pgn, language="plaintext")
        st.balloons()

        try:
            os.remove(temp_video_path)
        except:
            pass


if __name__=="__main__":
    main()

if __name__=="__main__":
    main()
