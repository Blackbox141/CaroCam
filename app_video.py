import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import os
import chess
import chess.pgn
import tempfile
from PIL import Image
import datetime

# Get the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative paths to the models for detection
piece_model_path = os.path.join(BASE_DIR, 'Figure Detection', 'runs', 'detect', 'yolov8-chess28', 'weights', 'best.pt')
corner_model_path = os.path.join(BASE_DIR, 'Corner Detection', 'runs', 'detect', 'yolov8n_corner12', 'weights', 'best.pt')
clock_model_path = os.path.join(BASE_DIR, 'Clock Detection 1', 'runs', 'detect', 'yolov8-chess3', 'weights', 'best.pt')

# Load the YOLO models with the relative paths
if os.path.exists(piece_model_path) and os.path.exists(corner_model_path) and os.path.exists(clock_model_path):
    piece_model = YOLO(piece_model_path)
    corner_model = YOLO(corner_model_path)
    clock_model = YOLO(clock_model_path)
else:
    st.error(f"Model file not found. Check the paths:\nPiece model: {piece_model_path}\nCorner model: {corner_model_path}\nClock model: {clock_model_path}")
    st.stop()

# FEN mapping for pieces
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

# Fixed values for correction vector
PERCENT_AB = 0.17  # Example value: 17% of the distance AB
PERCENT_BC = -0.07  # Example value: -7% of the distance BC

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

def detect_pieces(image):
    results = piece_model.predict(image, conf=0.1, iou=0.3, imgsz=1400)
    class_names = piece_model.model.names  # Access class names
    result = results[0]

    midpoints = []
    labels = []
    confidences = []
    boxes = result.boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls.cpu().numpy()[0])
        label = class_names[cls_id]
        conf = box.conf.cpu().numpy()[0]  # Confidence Score

        # Only proceed if confidence is above a threshold
        if conf < 0.5:  # Adjust threshold as needed
            continue

        # Calculate the midpoint of the lower half of the bounding box
        mid_x = x1 + (x2 - x1) / 2
        mid_y = y1 + (y2 - y1) * 0.75  # Middle of the lower half

        midpoints.append([mid_x, mid_y])
        labels.append(label)
        confidences.append(conf)

        # Draw the bounding box and label with confidence score on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        text = f"{label}: {conf*100:.1f}%"
        cv2.putText(image, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    return np.array(midpoints), labels, confidences, image

def detect_corners(image):
    results = corner_model(image)
    class_names = corner_model.model.names  # Access class names
    points = {}

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        cls_id = int(box.cls.cpu().numpy()[0])
        label = class_names[cls_id]

        if label == 'A':
            points["A"] = np.array([center_x, center_y])
        elif label == 'B':
            points["B"] = np.array([center_x, center_y])
        elif label == 'C':
            points["C"] = np.array([center_x, center_y])

        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(image, label, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    if len(points) != 3:
        return None, image

    return points, image

def detect_player_turn(image):
    results = clock_model.predict(image, conf=0.1, iou=0.3, imgsz=1400)
    class_names = clock_model.model.names  # Access class names
    result = results[0]
    boxes = result.boxes

    labels = []
    confidences = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls.cpu().numpy()[0])
        label = class_names[cls_id]
        conf = box.conf.cpu().numpy()[0]

        # Only proceed if confidence is above a threshold
        if conf < 0.5:  # Adjust threshold as needed
            continue

        labels.append(label)

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    # Determine who is to move based on labels
    if 'left' in labels and 'right' not in labels:
        player_turn = 'left'
    elif 'right' in labels and 'left' not in labels:
        player_turn = 'right'
    else:
        player_turn = None  # Require user input or hold

    if 'hold' in labels:
        player_turn = 'hold'

    return player_turn, image

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
    top_points = sorted(points[:2], key=lambda x: x[0])  # A and D
    bottom_points = sorted(points[2:], key=lambda x: x[0])  # B and C
    A_sorted, D_sorted = top_points
    B_sorted, C_sorted = bottom_points
    return np.array([A_sorted, B_sorted, C_sorted, D_sorted], dtype=np.float32)

def generate_fen_from_board(midpoints, labels, grid_size=8, player_to_move='w'):
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

    fen_string = '/'.join(fen_rows) + f" {player_to_move} - - 0 1"
    return fen_string

def get_move_between_positions(previous_fen, current_fen):
    previous_board = chess.Board(previous_fen)
    current_board = chess.Board(current_fen)

    move_made = None

    for move in previous_board.legal_moves:
        board_copy = previous_board.copy()
        board_copy.push(move)
        if board_copy.board_fen() == current_board.board_fen():
            return move  # Move found

    return None  # No move found

def save_game_to_pgn(moves, starting_fen):
    game = chess.pgn.Game()
    game.setup(chess.Board(starting_fen))
    node = game

    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        node = node.add_variation(move)

    pgn_string = str(game)
    return pgn_string

def main():
    st.title("Schachspiel Analyse aus Video mit Fortschrittsanzeige")

    # Upload video
    uploaded_file = st.file_uploader("Lade ein Video des Schachspiels hoch", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Create temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        # Get frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize progress bar
        progress_bar = st.progress(0)

        # Initialize variables
        previous_player_turn = None
        previous_player_turn_stable = None
        player_turn_buffer = []
        previous_fen = None
        move_list = []
        fen_list = []
        game_started = False
        starting_position = None
        white_side = None
        user_white_side = None
        game_over = False
        frame_count = 0

        current_player = 'w'  # Start with white

        # FEN buffer for stability check
        fen_buffer = []
        FEN_STABILITY_THRESHOLD = 3
        PLAYER_TURN_STABILITY_THRESHOLD = 5

        # Single board detection
        corners_detected = False

        while not corners_detected:
            ret, frame = cap.read()
            if not ret:
                st.error("Fehler beim Lesen des Videos oder keine weiteren Frames verfügbar.")
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Step 1: Detect corners
            detected_points, corner_image = detect_corners(frame_rgb)

            if detected_points is None:
                st.image(corner_image, caption='Eckpunkte nicht erkannt, nächstes Frame wird analysiert.', use_column_width=True)
                continue
            else:
                corners_detected = True
                st.image(corner_image, caption='Erkannte Eckpunkte', use_column_width=True)
                A = detected_points["A"]
                B = detected_points["B"]
                C = detected_points["C"]

        # Step 2: Calculate point D and perspective transformation
        D_calculated = calculate_point_D(A, B, C)
        D_corrected = adjust_point_D(A, B, C, D_calculated, PERCENT_AB, PERCENT_BC)
        sorted_points = sort_points(A, B, C, D_corrected.astype(int))
        M = cv2.getPerspectiveTransform(sorted_points, np.array([
            [0, 0],
            [0, 799],
            [799, 799],
            [799, 0]
        ], dtype=np.float32))

        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Process video
        while cap.isOpened() and not game_over:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(min(progress, 1.0))

            if frame_count % frame_interval != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = frame_rgb.copy()

            # Step 1: Detect player turn
            player_turn, clock_image = detect_player_turn(display_frame)

            # Update player turn buffer
            if player_turn is not None and player_turn != 'hold':
                player_turn_buffer.append(player_turn)
                if len(player_turn_buffer) > PLAYER_TURN_STABILITY_THRESHOLD:
                    player_turn_buffer.pop(0)
                if len(player_turn_buffer) == PLAYER_TURN_STABILITY_THRESHOLD and all(pt == player_turn_buffer[0] for pt in player_turn_buffer):
                    # Player turn is stable
                    if previous_player_turn_stable != player_turn_buffer[0]:
                        previous_player_turn_stable = player_turn_buffer[0]
                        st.write(f"Stabiler Uhrwechsel erkannt: {previous_player_turn} -> {player_turn_buffer[0]}")

                        # Determine current player
                        if previous_player_turn == 'left' and player_turn_buffer[0] == 'right':
                            current_player = 'b'
                        elif previous_player_turn == 'right' and player_turn_buffer[0] == 'left':
                            current_player = 'w'
                        elif previous_player_turn is None:
                            if player_turn_buffer[0] == 'left':
                                current_player = 'w'
                            elif player_turn_buffer[0] == 'right':
                                current_player = 'b'

                        previous_player_turn = player_turn_buffer[0]

                        # Display clock image
                        st.image(clock_image, caption=f'Analysiertes Frame {frame_count} - Schachuhr', use_column_width=True)

                        # Step 2: Detect pieces
                        piece_midpoints, piece_labels, piece_confs, piece_image = detect_pieces(display_frame)

                        # Display piece detection image
                        st.image(piece_image, caption=f'Analysiertes Frame {frame_count} - Figuren', use_column_width=True)

                        # Transform piece coordinates
                        if piece_midpoints.shape[0] > 0:
                            ones = np.ones((piece_midpoints.shape[0], 1))
                            piece_midpoints_homogeneous = np.hstack([piece_midpoints, ones])
                            transformed_midpoints = M @ piece_midpoints_homogeneous.T
                            transformed_midpoints /= transformed_midpoints[2, :]
                            transformed_midpoints = transformed_midpoints[:2, :].T
                        else:
                            transformed_midpoints = np.array([]).reshape(0, 2)

                        # Determine white's side
                        if white_side is None:
                            fen_found = False
                            for side in ["Links", "Rechts"]:
                                if side == "Links":
                                    rotated_midpoints = transformed_midpoints.copy()
                                elif side == "Rechts":
                                    rotated_midpoints = np.zeros_like(transformed_midpoints)
                                    rotated_midpoints[:, 0] = 800 - transformed_midpoints[:, 0]
                                    rotated_midpoints[:, 1] = 800 - transformed_midpoints[:, 1]

                                current_fen = generate_fen_from_board(rotated_midpoints, piece_labels, player_to_move=current_player)
                                st.write(f"**Aktuelle FEN-Notation ({side}):** {current_fen}")

                                if current_fen.startswith(STARTING_FEN):
                                    game_started = True
                                    starting_position = current_fen
                                    white_side = side
                                    fen_found = True
                                    st.write(f"Weiß wurde auf der Seite '{white_side}' erkannt.")
                                    break

                            if not fen_found:
                                if user_white_side is None:
                                    st.write("Bitte wählen Sie, auf welcher Seite Weiß spielt:")
                                    user_white_side = st.selectbox("Weiß spielt auf:", ("Links", "Rechts"))
                                    white_side = user_white_side
                                    st.write(f"Weiß wurde auf der Seite '{white_side}' festgelegt.")
                                else:
                                    white_side = user_white_side
                        else:
                            if white_side == "Links":
                                rotated_midpoints = transformed_midpoints.copy()
                            elif white_side == "Rechts":
                                rotated_midpoints = np.zeros_like(transformed_midpoints)
                                rotated_midpoints[:, 0] = 800 - transformed_midpoints[:, 0]
                                rotated_midpoints[:, 1] = 800 - transformed_midpoints[:, 1]

                            current_fen = generate_fen_from_board(rotated_midpoints, piece_labels, player_to_move=current_player)
                            st.write(f"**Aktuelle FEN-Notation (Weiß spielt '{white_side}'):** {current_fen}")

                        # Update FEN buffer
                        fen_buffer.append(current_fen)
                        if len(fen_buffer) > FEN_STABILITY_THRESHOLD:
                            fen_buffer.pop(0)

                        # Check if FEN is stable
                        if len(fen_buffer) == FEN_STABILITY_THRESHOLD and all(fen == fen_buffer[0] for fen in fen_buffer):
                            if previous_fen != fen_buffer[0]:
                                st.write("Zug erkannt aufgrund von FEN-Änderung.")

                                # Determine the move
                                if previous_fen is not None:
                                    move = get_move_between_positions(previous_fen, fen_buffer[0])
                                    if move is not None:
                                        move_list.append(move.uci())
                                        st.write(f"Erkannter Zug: {move.uci()}")
                                    else:
                                        st.write("Genauer Zug konnte nicht ermittelt werden.")
                                        move_list.append("Unbekannter Zug")

                                if len(fen_list) == 0 or fen_buffer[0] != fen_list[-1]:
                                    fen_list.append(fen_buffer[0])

                                previous_fen = fen_buffer[0]

                                # Check for checkmate
                                board = chess.Board(fen_buffer[0])
                                if board.is_checkmate():
                                    st.write("**Schachmatt!**")
                                    game_over = True

                        else:
                            st.write("FEN ist noch nicht stabil.")

                else:
                    st.write("Spielerzug ist noch nicht stabil.")

            else:
                pass  # Do nothing

        cap.release()
        st.write("Videoverarbeitung abgeschlossen.")

        # Display FEN list
        if fen_list:
            st.write("**Liste der erkannten FENs:**")
            for idx, fen in enumerate(fen_list):
                st.write(f"Position {idx+1}: {fen}")

        # Generate PGN if game started
        if game_started and move_list:
            pgn_string = save_game_to_pgn(move_list, starting_position)
            st.write("**PGN des Spiels:**")
            st.code(pgn_string)
        else:
            st.write("Keine Züge erkannt oder Spiel hat nicht von der Grundstellung begonnen.")

    else:
        st.write("Bitte lade ein Video des Schachspiels hoch.")

if __name__ == "__main__":
    main()
