import cv2
import mediapipe as mp
import time
import tkinter as tk
from tkinter import Label
from tkinter import Listbox, END
from PIL import Image, ImageTk, ImageDraw, ImageFont
import math
import numpy as np
import random
from pygame import mixer
import openai
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask is running!"

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

EAR_THRESHOLD = 0.30  
blink_count = 0  
score = 0 
is_running = False  
current_phase = 0  
phase_start_time = None  
total_time_limit = 180 
start_time = None  
expected_mouth_state = None  
previous_hand_position = None 
MOTION_THRESHOLD = 10  
previous_blink_state = "open"
current_ear = 0.0
openai.api_key = "sk-O7eiZZSPb-pEme3DoUu7HuRnN7Y9T4WCftWUIvXlsxT3BlbkFJcGmAj29NshozrMekwSOPjZ6B7Ki0rIyL0Y6t-uHlMA"
crown_image = cv2.imread("oukan.png", cv2.IMREAD_UNCHANGED)
mp_face_detection = mp.solutions.face_detection 
mp_face_mesh_instance = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
right_brush_count = 0
left_brush_count = 0
front_brush_count = 0


 
OMIKUJI_RESULTS = ["大吉", "中吉", "小吉", "吉", "末吉", "凶"]

mp_face_mesh_instance = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_hands_instance = mp_hands.Hands(max_num_hands=1)

def process_frame(frame):
    global mp_face_mesh_instance, mp_hands_instance

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print("Debug: process_frame() が呼び出されました")

    if mp_face_mesh_instance is None or mp_hands_instance is None:
        print("Error: MediaPipe インスタンスが None です！")
        return None, None  

    try:
        face_results = mp_face_mesh_instance.process(rgb_frame)
    except Exception as e:
        print(f"Error: mp_face_mesh_instance.process() でエラー発生: {e}")
        face_results = None  

    try:
        hand_results = mp_hands_instance.process(rgb_frame)
    except Exception as e:
        print(f"Error: mp_hands_instance.process() でエラー発生: {e}")
        hand_results = None

    return face_results, hand_results



phases = [
    {"mode": "普通の歯磨き", "description": "歯を磨きましょう。", "duration": 30, "bgm": "ふつうのはみがき.mp3"},
    {"mode": "い歯磨き", "description": "口を『い』の形にして歯を磨きましょう。", "expected_mouth": "i", "duration": 30, "bgm": "い.mp3"},
    {"mode": "普通の歯磨き", "description": "歯を磨きましょう。", "duration": 20, "bgm": "ふつうのはみがき.mp3"},
    {"mode": "瞬きモード", "description": "手を画面からはずし、瞬きに集中して得点を稼ぎましょう。", "duration": 10, "bgm": "まばたき.mp3"},
    {"mode": "普通の歯磨き", "description": "歯を磨きましょう。", "duration": 20, "bgm": "ふつうのはみがき.mp3"},
    {"mode": "あ歯磨き", "description": "口を『あ』の形にして歯を磨きましょう。", "expected_mouth": "a", "duration": 20, "bgm": "あ.mp3"},
    {"mode": "普通の歯磨き", "description": "歯を磨きましょう。", "duration": 15, "bgm": "ふつうのはみがき.mp3"},
    {"mode": "おみくじモード", "description": "手を画面からはずし、瞬き25回でおみくじ結果を見ましょう！", "duration": None, "bgm": "まばたき.mp3"},
    {"mode": "普通の歯磨き", "description": "時間が終わるまで歯を磨き続けましょう。", "duration": None, "bgm": "ふつうのはみがき.mp3"},
    
]


mixer.init()


def calculate_ear(landmarks, indices):

    vertical_1 = math.dist(
        [landmarks[indices[1]].x, landmarks[indices[1]].y],
        [landmarks[indices[5]].x, landmarks[indices[5]].y],
    )
    vertical_2 = math.dist(
        [landmarks[indices[2]].x, landmarks[indices[2]].y],
        [landmarks[indices[4]].x, landmarks[indices[4]].y],
    )
    horizontal = math.dist(
        [landmarks[indices[0]].x, landmarks[indices[0]].y],
        [landmarks[indices[3]].x, landmarks[indices[3]].y],
    )
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def detect_mouth_state(landmarks, width, height):
    if landmarks is None or len(landmarks) < 15:  
        print(f"Error: ランドマークが不足しています (現在の長さ: {len(landmarks)})")
        return "unknown"  

    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    mouth_width = np.linalg.norm(np.array(landmarks[78]) - np.array(landmarks[308]))
    mouth_height = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip))

    # 口の状態
    if mouth_height < 0.01 * height:
        return "closed"
    elif mouth_height / mouth_width > 0.5:
        return "a"
    elif mouth_width > 0 and mouth_height / mouth_width <= 0.5:
        return "i"
    else:
        return "unknown"


def calculate_index_pinky_vector(pinky_pip, index_mcp):
    vector = np.array([index_mcp[0] - pinky_pip[0], pinky_pip[1] - index_mcp[1]])  
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi  

    angle = abs(angle)  
    if angle > 180:
        angle = 360 - angle  

    return vector, angle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles  
def hand_movement_detected(current_hand_position):
 
    global previous_hand_position

    if previous_hand_position is None:
        previous_hand_position = current_hand_position
        return False

    movement = np.linalg.norm(np.array(current_hand_position) - np.array(previous_hand_position))
    previous_hand_position = current_hand_position
    return movement > MOTION_THRESHOLD

def calculate_index_pinky_angle(pinky_pip, index_mcp):
    # 小指の中節（ランドマーク17）と人差し指の付け根（ランドマーク6）の角度を計算
    vector = np.array([index_mcp[0] - pinky_pip[0], pinky_pip[1] - index_mcp[1]])  
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi  
    angle = abs(angle)  
    if angle > 180:
        angle = 360 - angle  

    return vector, angle


def calculate_index_angle(landmark5, landmark6):
    # 人差し指のPIP（5）とMCP（6）の角度を計算
    vector = np.array([landmark6[0] - landmark5[0], landmark6[1] - landmark5[1]])
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi  
    return abs(angle)  

    angle = abs(angle)  
    if angle > 180:
        angle = 360 - angle  

    return vector, angle


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  
mp_drawing_styles = mp.solutions.drawing_styles  
def hand_movement_detected(current_hand_position):
 
    global previous_hand_position

    if previous_hand_position is None:
        previous_hand_position = current_hand_position
        return False

    movement = np.linalg.norm(np.array(current_hand_position) - np.array(previous_hand_position))
    previous_hand_position = current_hand_position
    return movement > MOTION_THRESHOLD


def analyze_brushing(hand_landmarks):
    global score
    global angle_label_5_6, angle_label_17_6, phase_status_label
    global right_brush_count, left_brush_count, front_brush_count  

    if not hand_landmarks:
        print("Debug: 手のランドマークが検出されていません。")
        return

    brushing_side = "未判定"

    # 人差し指（5）と（6）の角度
    index_pip = (hand_landmarks[5][0], hand_landmarks[5][1])
    index_mcp = (hand_landmarks[6][0], hand_landmarks[6][1])
    angle_5_6 = calculate_index_angle(index_pip, index_mcp)

    angle_label_5_6.config(text=f"人差し指の角度: {angle_5_6:.2f} 度")

    # 親指の先端（4）と小指の付け根（17）の角度
    thumb_tip_xy = (hand_landmarks[4][0], hand_landmarks[4][1])  # 親指
    pinky_base_xy = (hand_landmarks[17][0], hand_landmarks[17][1])  # 小指

    _, angle_4_17 = calculate_thumb_pinky_angle(thumb_tip_xy, pinky_base_xy)

    angle_label.config(text=f"親指と小指の角度: {angle_4_17:.2f} 度")

    # 小指の中節（17）と人差し指の付け根（6）の角度
    pinky_pip = (hand_landmarks[17][0], hand_landmarks[17][1])
    index_mcp = (hand_landmarks[6][0], hand_landmarks[6][1])
    _, angle_17_6 = calculate_index_pinky_angle(pinky_pip, index_mcp)

    angle_label_17_6.config(text=f"小指と人差し指の角度: {angle_17_6:.2f} 度")

    if angle_17_6 >= 160:
        brushing_side = "前を磨いています"
        front_brush_count += 1
    elif angle_17_6 <= 85:
        brushing_side = "右を磨いています"
        right_brush_count += 1
    else:
        brushing_side = "左を磨いています"
        left_brush_count += 1

    print(f"ランドマーク4→17の角度: {angle_4_17:.2f} 度 - {brushing_side}")

    phase_status_label.config(text=f"歯磨き中！{brushing_side}")
    score += 1

def update_brushing_stats():
    #歯磨きの割合を表示する
    total = right_brush_count + left_brush_count + front_brush_count
    if total == 0:
        right_percent = left_percent = front_percent = 0
    else:
        right_percent = (right_brush_count / total) * 100
        left_percent = (left_brush_count / total) * 100
        front_percent = (front_brush_count / total) * 100

    brushing_stats_label.config(
        text=f"右 {right_percent:.1f}%  左 {left_percent:.1f}%  前 {front_percent:.1f}%"
    )

    root.after(1000, update_brushing_stats)  


def calculate_thumb_pinky_angle(thumb_tip, pinky_base):
    # 親指の先端（ランドマーク4）と小指の付け根（ランドマーク17）の角度を計算
    vector = np.array([pinky_base[0] - thumb_tip[0], thumb_tip[1] - pinky_base[1]])  
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi  

    angle = abs(angle)  
    if angle > 180:
        angle = 360 - angle  

    return vector, angle



def analyze_blink(face_landmarks):
    global blink_count, score, previous_blink_state, current_ear

    left_eye_indices = [33, 159, 145, 133, 153, 144]
    right_eye_indices = [263, 386, 374, 362, 380, 373]

    left_ear = calculate_ear(face_landmarks, left_eye_indices)
    right_ear = calculate_ear(face_landmarks, right_eye_indices)
    ear = (left_ear + right_ear) / 2.0

    current_ear = ear
    ear_label.config(text=f"EAR: {current_ear:.2f}")

    if ear < EAR_THRESHOLD:
        blink_state = "closed"
    else:
        blink_state = "open"

    
    if previous_blink_state == "closed" and blink_state == "open":
        if phases[current_phase]["mode"] == "おみくじモード":
            blink_count += 1
            score += 1
            phase_status_label.config(text=f"瞬き成功！スコア: {score} 瞬き回数: {blink_count}")
            update_blink_count_label() 
        elif phases[current_phase]["mode"] == "瞬きモード":
            blink_count += 1
            score += 1
            phase_status_label.config(text=f"瞬き成功！スコア: {score} 瞬き回数: {blink_count}")
            update_blink_count_label() 

        
        if phases[current_phase]["mode"] == "おみくじモード" and blink_count == 25:
            display_omikuji_result()
            blink_count = 0  

    previous_blink_state = blink_state

def update_blink_count_label():
    """瞬き回数ラベルをリアルタイム更新"""
    blink_count_label.config(text=f"瞬き回数: {blink_count}回")



def display_omikuji_result():

    global current_phase, phase_start_time

    result = random.choice(OMIKUJI_RESULTS)
    omikuji_label.config(text=f"おみくじ結果: {result}")


    current_phase += 1
    phase_start_time = time.time()


def draw_phase_status(canvas, current_phase, phases):

    canvas.delete("all")  

    x, y = 10, 20  
    rect_width, rect_height = 200, 50 
    vertical_spacing = 5
 

    for i, phase in enumerate(phases):
   
        mode = phase.get("mode", "未設定")
        duration = phase.get("duration", "∞")

        if isinstance(duration, int):
            duration_text = f"{duration}秒"
        else:
            duration_text = "∞"

      
        if i == current_phase:
            fill_color = "blue"
            text_color = "white"
        elif i == current_phase + 1:
            fill_color = "green"  
            text_color = "black"
        else:
            fill_color = "gray"
            text_color = "black"

          
        canvas.create_rectangle(
            x, y + i * (rect_height + vertical_spacing),
            x + rect_width, y + rect_height + i * (rect_height + vertical_spacing),
            fill=fill_color, outline="black"
        )

        canvas.create_text(
            x + rect_width / 2, y + 15 + i * (rect_height + vertical_spacing),
            text=mode, fill=text_color, font=("Arial", 14, "bold")  
        )

        canvas.create_text(
            x + rect_width / 2, y + 35 + i * (rect_height + vertical_spacing),
            text=duration_text, fill=text_color, font=("Arial", 12)  
        )
          
        if i == current_phase + 1:
            canvas.create_text(
                x + rect_width - 40, y + 10 + i * (rect_height + vertical_spacing),
                text="NEXT",
                fill="red",
                font=("Arial", 14, "bold")
            )


def place_crown_on_face(frame, face_bbox):
    x = int(face_bbox.xmin * frame.shape[1])
    y = int(face_bbox.ymin * frame.shape[0])
    width = int(face_bbox.width * frame.shape[1])
    height = int(face_bbox.height * frame.shape[0])

    # 王冠のサイズと位置を調整
    crown_width = int(width * 1.5)
    crown_height = int(crown_width * crown_image.shape[0] / crown_image.shape[1])
    crown_x1 = x - int((crown_width - width) / 2)
    crown_y1 = y - crown_height
    crown_x2 = crown_x1 + crown_width
    crown_y2 = crown_y1 + crown_height
    resized_crown = cv2.resize(crown_image, (crown_width, crown_height))

    # フレームに王冠を合成する
    for i in range(crown_height):
        for j in range(crown_width):
            if 0 <= crown_x1 + j < frame.shape[1] and 0 <= crown_y1 + i < frame.shape[0]:
                alpha = resized_crown[i, j, 3] / 255.0  
                if alpha > 0:  
                    frame[crown_y1 + i, crown_x1 + j] = (
                        alpha * resized_crown[i, j, :3] + (1 - alpha) * frame[crown_y1 + i, crown_x1 + j]
                    )
    return frame
    
    

def update_phase(frame):
    global current_phase, phase_start_time, score, is_running, expected_mouth_state

    if frame is None or frame.size == 0:
        print("Error: update_phaseに渡されたフレームが無効です。")
        return frame  

    processed_frame = frame.copy()

    try:
        if current_phase < len(phases):
            current_time = time.time()
            elapsed_phase_time = current_time - phase_start_time

            if phases[current_phase]["duration"] is not None and elapsed_phase_time >= phases[current_phase]["duration"]:
                current_phase += 1
                phase_start_time = current_time

                if current_phase >= len(phases):
                    phase_status_label.config(text="終了ー！")
                    stop_game()
                    return processed_frame

                draw_phase_status(phase_canvas, current_phase, phases)
                play_bgm_for_phase()

            phase = phases[current_phase]
            mode_label.config(text=f"現在のモード: {phase['mode']}")
            instruction_label.config(text=f"説明: {phase['description']}")

            expected_mouth_state = phase.get("expected_mouth", None)

        if phase["mode"] in ["普通の歯磨き", "い歯磨き", "あ歯磨き"]:
            with mp_hands.Hands(max_num_hands=1) as hands:
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    hand_landmarks_list = [
                        [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                        for hand in results.multi_hand_landmarks
                    ]
                    analyze_brushing(hand_landmarks_list[0])

                with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    face_results = face_mesh.process(rgb_frame)
                if face_results.multi_face_landmarks:
                 for face_landmarks in face_results.multi_face_landmarks:
                  landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                 current_mouth_state = detect_mouth_state(
                 landmarks, processed_frame.shape[1], processed_frame.shape[0]
            )

                if phase["mode"] in ["い歯磨き", "あ歯磨き"]:
                                if current_mouth_state == expected_mouth_state:
                                    analyze_brushing(hand_landmarks_list[0])
                                else:
                                    phase_status_label.config(text=f"口の形が「{expected_mouth_state}」ではありません")

            processed_frame = add_crown_to_face(processed_frame)
        if phase["mode"] == "瞬きモード":
         with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh, mp_hands.Hands(max_num_hands=1) as hands:
          rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        # 瞬き処理
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                analyze_blink(face_landmarks.landmark)  
                print("Debug: 瞬き処理が実行されました。")

        # 歯磨き（手の動き）処理
        if hand_results.multi_hand_landmarks:
            hand_landmarks_list = [
                [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                for hand in hand_results.multi_hand_landmarks
            ]
            for hand_landmarks in hand_landmarks_list:
                analyze_brushing(hand_landmarks)  
                print("Debug: 歯磨き処理が実行されました。")
        else:
            print("Debug: 手のランドマークが検出されませんでした。")

        if phase["mode"] == "おみくじモード":
         phase_status_label.config(text=f"瞬き回数: {blink_count}")
        if blink_count >= 20:
         display_omikuji_result()


    except cv2.error as e:
        print(f"OpenCVエラー: {e}")
        return processed_frame
    except Exception as e:
        print(f"不明なエラー: {e}")
        return processed_frame

    return processed_frame



def add_crown_to_face(frame):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                frame = place_crown_on_face(frame, detection.location_data.relative_bounding_box)
    return frame




def update_camera():
    global is_running

    if not is_running:
        return

    retry_count = 5  
    frame = None

    for attempt in range(retry_count):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        print(f"Warning: フレーム取得に失敗（試行 {attempt + 1}/{retry_count}）")
    else:
        feedback_label.config(text="カメラの映像を取得できません。")
        print("Error: カメラの映像を取得できませんでした。")
        return

    print(f"Debug: フレーム取得成功 - サイズ: {frame.shape if frame is not None else 'None'}")

    try:
        frame = cv2.flip(frame, 1)

        if frame is None or frame.size == 0:
            print("Warning: フレームが空です。処理をスキップします。")
            feedback_label.config(text="フレームが空です。処理をスキップします。")
            return

        height, width, _ = frame.shape

        # 十字を描く
        line_color = (192, 192, 192)  
        thickness = 1  
        center_x, center_y = width // 2, height // 2 + 40

        cv2.line(frame, (center_x - 40, center_y), (center_x + 40, center_y), line_color, thickness)
        cv2.line(frame, (center_x, center_y - 40), (center_x, center_y + 40), line_color, thickness)

        frame = update_phase(frame)

        if ret and frame is not None:
            print(f"デバッグ: フレームサイズ - {frame.shape}")  
        else:
            print("エラー: フレームが取得できませんでした。")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape

        hand_landmarks_list = []
        landmarks = None
        with mp_hands.Hands(max_num_hands=1) as hands, mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
            hand_results = hands.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    hand_landmarks_list = [
                        (lm.x * width, lm.y * height, lm.z) for lm in hand_landmarks.landmark
                    ]
                print(f"Debug: 手のランドマーク検出 - {len(hand_landmarks_list)}手")
            else:
                print("Debug: 手のランドマークが検出されませんでした。")
                landmarks = []  

  
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    landmarks = [(lm.x * width, lm.y * height, lm.z) for lm in face_landmarks.landmark]
                    current_mouth_state = detect_mouth_state(landmarks, width, height)
                    print(f"Debug: 顔のランドマーク検出 - {len(landmarks)}ポイント")
            else:
                print("Debug: 顔のランドマークが検出されませんでした。")

                 
            if face_results.multi_face_landmarks:
             for face_landmarks in face_results.multi_face_landmarks:
              if face_landmarks and hasattr(face_landmarks, "landmark"):  
                analyze_blink(face_landmarks.landmark)
                print(f"Debug: 顔のランドマーク検出 - {len(face_landmarks.landmark)}ポイント")
              else:
               print("Debug: face_landmarksが無効か、landmark属性がありません。")
             else:
              print("Debug: 顔のランドマークが検出されませんでした。")


            if landmarks:
             current_mouth_state = detect_mouth_state(landmarks, width, height)
            else:
             current_mouth_state = "unknown"  
             print("Debug: ランドマークがないため、口の状態を判定できませんでした。")
                

    except cv2.error as e:
        print(f"OpenCVエラー: {e}")
        feedback_label.config(text=f"OpenCVエラー: {e}")
        return
    except Exception as e:
        print(f"不明なエラー: {e}")
        feedback_label.config(text=f"不明なエラー: {e}")
        return

    current_mouth_state = detect_mouth_state(landmarks, width, height)
    if current_phase < len(phases):
                    phase = phases[current_phase]
                    if phase["mode"] == "瞬きモード" or phase["mode"] == "おみくじモード":
                        analyze_blink(face_landmarks.landmark)
                        analyze_brushing(hand_landmarks_list)  
                    elif phase["mode"] == "普通の歯磨き":
                        analyze_brushing(hand_landmarks_list)
                    elif phase["mode"] in ["い歯磨き", "あ歯磨き"]:
                        if current_mouth_state == expected_mouth_state:
                            analyze_brushing(hand_landmarks_list)
                        else:
                            phase_status_label.config(text=f"口の形が「{expected_mouth_state}」ではありません")
    
    try:
        new_width = 600  
        new_height = 450  
        frame_pil = Image.fromarray(rgb_frame)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)
        camera_label.imgtk = frame_tk
        camera_label.config(image=frame_tk)
    except Exception as e:
        print(f"Pillowエラー: {e}")
        feedback_label.config(text=f"Pillowエラー: {e}")
        return

    score_label.config(text=f"スコア: {score}")

    root.after(50, update_camera)



def update_timers():
    if not is_running:
        return

    current_time = time.time()
    elapsed_time = int(current_time - start_time)
    remaining_time = max(0, total_time_limit - elapsed_time)


    remaining_time_label.config(text=f"全体残り時間: {remaining_time // 60}:{remaining_time % 60:02}")


    if phases[current_phase]["duration"] is not None:
        elapsed_phase_time = current_time - phase_start_time
        phase_remaining_time = max(0, int(phases[current_phase]["duration"] - elapsed_phase_time))
        phase_time_label.config(text=f"フェーズ残り時間: {phase_remaining_time} 秒")
    else:

        phase_time_label.config(text=f"フェーズ残り時間: {remaining_time // 60}:{remaining_time % 60:02}")

        if remaining_time <= 0:
            phase_status_label.config(text="終了ー！")
            stop_game()

    root.after(1000, update_timers)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: カメラが起動できませんでした。")
    exit()

root = tk.Tk()
root.title("hamigaking")
root.configure(bg="lightblue")


start_button_image_original = Image.open("button.png")
stop_button_image_original = Image.open("button.png")
start_button_image_resized = start_button_image_original.resize((120, 60), Image.Resampling.LANCZOS)
stop_button_image_resized = stop_button_image_original.resize((120, 60), Image.Resampling.LANCZOS)


def countdown(seconds):
   
    global is_running
    is_running = False  
    for i in range(seconds, 0, -1):
        phase_status_label.config(text=f"カウントダウン: {i} 秒")
        root.update()
        mixer.music.load("カウントダウン電子音.mp3")  
        mixer.music.play()
        time.sleep(1)
    mixer.music.stop()
    phase_status_label.config(text="スタート！")
    start_game()  

def log_brushing_data():
    # print("現在のデータ:", brushing_data)
    root.after(1000, log_brushing_data)  


def start_game():
    global is_running, score, current_phase, phase_start_time, start_time, brushing_data
    is_running = True
    score = 0
    current_phase = 0
    phase_start_time = time.time()
    start_time = time.time()
    right_brush_count = 0
    left_brush_count = 0
    front_brush_count = 0
    blink_count = 0

    update_brushing_stats()
    update_blink_count_label()
    
    mode_label.config(text=f"現在のモード: {phases[current_phase]['mode']}")  
    instruction_label.config(text=f"説明: {phases[current_phase]['description']}")  
    phase_time_label.config(text=f"フェーズ残り時間: {phases[current_phase]['duration']} 秒")  
    phase_status_label.config(text="状況: ゲーム開始！")  
    omikuji_label.config(text="おみくじ結果: ")  
    
    draw_phase_status(phase_canvas, current_phase, phases)

    play_bgm_for_phase()

    ret, frame = cap.read()  
    if not ret:
        feedback_label.config(text="カメラの映像を取得できません")
        return

    frame = cv2.flip(frame, 1)
    update_phase(frame)  
    update_camera()  
    update_timers()  
    log_brushing_data()  

  

def calculate_rank(score):
    if score >= 1200:
        return "S"
    elif score >= 1000:
        return "A"
    elif score >= 800:
        return "B"
    elif score >= 600:
        return "C"
    elif score >= 400:
        return "D"
    else:
        return "E"

def generate_advice(score):
    rank = calculate_rank(score)  
    brushing_summary = f"今回の歯磨きでの得点は {score} 点です。ランクは {rank} です。"


    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは歯磨きのアドバイスを提供する歯科医です。"},
                {"role": "user", "content": f"{brushing_summary} この得点とランクに基づいて振り返りとアドバイスを提供してください。"}
            ],
            temperature=0.7, 
        )
        advice = response["choices"][0]["message"]["content"]
        return advice
    except Exception as e:
        print(f"OpenAI APIエラー: {e}")
        return "アドバイスを生成できませんでした。"

def stop_game():
 
    global is_running, score
    is_running = False
    mixer.music.stop()  
    phase_status_label.config(text="状況: ゲーム停止")
    mode_label.config(text="現在のモード: 待機中")  
    instruction_label.config(text="説明: -")  
    phase_time_label.config(text="フェーズ残り時間: -")  
    remaining_time_label.config(text="全体残り時間: 3:00")  

    blink_count = 0  
    current_phase = 0  
    draw_phase_status(phase_canvas, current_phase, phases)  

    rank = calculate_rank(score)
  
    advice = generate_advice(score)
    
    advice_window = tk.Toplevel(root)
    advice_window.title("歯磨きの振り返りとアドバイス")

    result_text = f"{score} ポイント！ランクは {rank}！"
    result_label = tk.Label(advice_window, text=result_text, font=("Arial", 14, "bold"), bg="white", wraplength=400)
    result_label.pack(padx=20, pady=10)

    advice_label = tk.Label(advice_window, text=f"AIのアドバイス:\n{advice}", font=("Arial", 12), bg="white", wraplength=400)
    advice_label.pack(padx=20, pady=20)



def add_text_to_image(image, text, font_path="C:/Windows/Fonts/meiryo.ttc", font_size=20, position=(10, 10), text_color=(255, 255, 255), outline_color=(0, 0, 0)):
 
    image_with_text = image.copy()
    draw = ImageDraw.Draw(image_with_text)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    
    for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((position[0] + offset[0], position[1] + offset[1]), text, font=font, fill=outline_color)
   
    draw.text(position, text, font=font, fill=text_color)
    return image_with_text


start_button_image_original = Image.open("button.png")  
stop_button_image_original = Image.open("button.png")    

start_button_image_resized = start_button_image_original.resize((120, 60), Image.Resampling.LANCZOS)
stop_button_image_resized = stop_button_image_original.resize((120, 60), Image.Resampling.LANCZOS)

start_button_image_with_text = ImageTk.PhotoImage(
    add_text_to_image(
        start_button_image_resized, 
        "スタート", 
        font_size=14, 
        position=(30, 20), 
        text_color=(0, 0, 0), 
        outline_color=(255, 255, 255)
    )
)

stop_button_image_with_text = ImageTk.PhotoImage(
    add_text_to_image(
        stop_button_image_resized, 
        "ストップ", 
        font_size=14, 
        position=(30, 20), 
        text_color=(0, 0, 0), 
        outline_color=(255, 255, 255)
    )
)

start_button = tk.Button(
    root, 
    image=start_button_image_with_text,  
    command=lambda: countdown(3),  
    bg="#00AEEF", 
    borderwidth=0
)


stop_button = tk.Button(
    root, 
    image=stop_button_image_with_text,  
    command=stop_game, 
    bg="#00AEEF", 
    borderwidth=0
)

start_button.grid(row=3, column=0, sticky="w", padx=(20, 10), pady=10)
stop_button.grid(row=4, column=0, sticky="w", padx=(20, 10), pady=10)



root.start_button_image = start_button_image_with_text
root.stop_button_image = stop_button_image_with_text

def play_bgm_for_phase():

    if current_phase >= len(phases):  
        return

    bgm_file = phases[current_phase].get("bgm")  
    if bgm_file:
        try:
            mixer.music.load(bgm_file)  
            mixer.music.play(-1)  
        except pygame.error as e:
            print(f"BGMファイルの読み込みエラー: {e}")



def draw_phase_status(canvas, current_phase, phases):
    canvas.delete("all")
    x, y = 10, 10
    rect_width, rect_height = 280, 60
    vertical_spacing = 15

    for i, phase in enumerate(phases):
        mode = phase.get("mode", "未設定")
        duration = phase.get("duration", "∞")

        if isinstance(duration, int):
            duration_text = f"{duration}秒"
        else:
            duration_text = "∞"

        fill_color = "gray"
        text_color = "black"

        if i == current_phase:
            fill_color = "blue"
            text_color = "white"
        elif i == current_phase + 1:
            fill_color = "green"

        canvas.create_rectangle(
            x, y + i * (rect_height + vertical_spacing),
            x + rect_width, y + rect_height + i * (rect_height + vertical_spacing),
            fill=fill_color, outline="black", tags=f"phase_{i}"
        )


        canvas.create_text(
            x + rect_width / 2, y + 15 + i * (rect_height + vertical_spacing),
            text=mode, fill=text_color, font=("Arial", 16), tags=f"phase_{i}"
        )

        canvas.create_text(
            x + rect_width / 2, y + 45 + i * (rect_height + vertical_spacing),
            text=duration_text, fill=text_color, font=("Arial", 14), tags=f"phase_{i}"
        )

        if i < len(phases) - 2:  # 最後の2つは固定
            canvas.tag_bind(f"phase_{i}", "<ButtonPress-1>", on_drag_start)
            canvas.tag_bind(f"phase_{i}", "<B1-Motion>", on_drag_motion)
            canvas.tag_bind(f"phase_{i}", "<ButtonRelease-1>", on_drag_drop)

dragging_phase = None  
dragging_index = None  

def on_drag_start(event):
    global dragging_phase, dragging_index
    canvas = event.widget
    x, y = event.x, event.y
    for i in range(len(phases) - 2):  # 最後の2つは固定
        bbox = canvas.bbox(f"phase_{i}")
        if bbox and bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            dragging_phase = phases[i]
            dragging_index = i
            break

def on_drag_motion(event):
    if dragging_phase is not None:
        canvas = event.widget
        canvas.delete("highlight")
        y = event.y

        # どのフェーズと入れ替えるか
        for i in range(len(phases) - 2):
            bbox = canvas.bbox(f"phase_{i}")
            if bbox and bbox[1] <= y <= bbox[3]:
                canvas.create_rectangle(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    outline="red", width=2, tags="highlight"
                )
                break

def on_drag_drop(event):
    global dragging_phase, dragging_index
    if dragging_phase is None or dragging_index is None:
        return

    canvas = event.widget
    y = event.y
    new_index = dragging_index

    for i in range(len(phases) - 2):
        bbox = canvas.bbox(f"phase_{i}")
        if bbox and bbox[1] <= y <= bbox[3]:
            new_index = i
            break

    if new_index != dragging_index:
        phases.insert(new_index, phases.pop(dragging_index))
    
    draw_phase_status(canvas, current_phase, phases)
    dragging_phase, dragging_index = None, None


phase_canvas = tk.Canvas(root, width=300, height=680, bg="#00AEEF", highlightthickness=0)
phase_canvas.grid(row=1, column=3, rowspan=5, padx=20, pady=5, sticky="n")


draw_phase_status(phase_canvas, current_phase, phases)


logo_image_path = "logo.png"
logo_image = Image.open(logo_image_path)
logo_image_resized = logo_image.resize((300, 100), Image.Resampling.LANCZOS)
logo_image_tk = ImageTk.PhotoImage(logo_image_resized)

logo_label = tk.Label(root, image=logo_image_tk, bg="#00AEEF")
logo_label.grid(row=0, column=1, columnspan=2, pady=10)

camera_label = Label(root, bg="lightblue")
camera_label.grid(row=1, column=1, rowspan=1, padx=(20, 50), pady=20, sticky="nsew")
root.columnconfigure(1, weight=2)  
root.rowconfigure(1, weight=2)    


draw_phase_status(phase_canvas, current_phase, phases)


phase_status_label = Label(root, text="状況: 待機中…", font=("Arial", 20), bg="lightblue")
phase_status_label.grid(row=4, column=1, columnspan=2, pady=0, sticky="ew")


mode_label = Label(root, text="現在のモード: 待機中", font=("Arial", 20), bg="lightblue")
mode_label.grid(row=2, column=1, columnspan=2, pady=0, sticky="ew")

instruction_label = Label(root, text="説明: -", font=("Arial", 20), bg="lightblue")
instruction_label.config(font=("Arial", 20), wraplength=600)
instruction_label.grid(row=3, column=1, columnspan=2, pady=0, sticky="ew")

score_label = tk.Label(root, text="得点: 0点", font=("Arial", 20,"bold"), bg="#00AEEF", fg="black")
score_label.grid(row=0, column=0, sticky="e", padx=(5, 5), pady=(2, 2))


blink_count_label = tk.Label(
    root, 
    text="瞬き回数:  0回",  
    font=("Arial", 18, "bold"),  
    bg="#00AEEF",
    fg="black"
)
blink_count_label.grid(row=1, column=0, sticky="w", padx=(20, 10), pady=(2, 2))


phase_time_label= tk.Label(
    root, 
   
    text="フェーズ残り時間: 未設定", 
    font=("Arial", 18), 
    bg="lightblue", 
    fg="red", 
    pady=3  
)

phase_time_label.grid(row=4, column=1, columnspan=2, pady=5, sticky="ew")

remaining_time_label = tk.Label(
    root, 
    text="全体残り時間: 3:00", 
    font=("Arial", 18),
    bg="lightblue", 
    fg="red", 
    pady=3 
)
remaining_time_label.grid(row=5, column=1, columnspan=2, pady=5, sticky="ew")  
root.columnconfigure(1, weight=1)  


phase_time_label.config(font=("Arial", 18), fg="red")
phase_time_label.grid(row=6, column=1, columnspan=2, pady=5, sticky="ew")

omikuji_label = tk.Label(root, text="おみくじ結果: ", font=("Arial", 18,"bold"), bg="#00AEEF", fg="black")
omikuji_label.grid(row=2, column=0, sticky="w", padx=(20, 10), )


feedback_label = tk.Label(root, text="", font=("Arial", 14), bg="lightblue")
feedback_label.grid(row=8, column=1, sticky="w")

ear_label = tk.Label(root, text="EAR: 0.00", font=("Arial", 14), bg="lightblue")
ear_label.grid(row=9, column=1, sticky="w")

root.protocol("WM_DELETE_WINDOW", lambda: [cap.release(), root.destroy()])


angle_label = tk.Label(root, text="親指と小指の角度: 0.00 度", font=("Arial", 14), bg="lightblue")
angle_label.grid(row=10, column=1, sticky="w")  

angle_label_5_6 = Label(root, text="人差し指の角度: 0.00 度", font=("Arial", 14), bg="lightblue")
angle_label_5_6.grid(row=11, column=1, sticky="w") 

angle_label_17_6 = tk.Label(root, text="小指と人差し指の角度: 0.00 度", font=("Arial", 14), bg="lightblue")
angle_label_17_6.grid(row=12, column=1, sticky="w")  

brushing_stats_label = tk.Label(
    root,
    text="右 0%  左 0%  前 0%",
    font=("Arial", 14),
    bg="lightblue",
    fg="black"
)
brushing_stats_label.grid(row=6, column=3, padx=10, pady=5, sticky="ew")

from threading import Thread

def run_flask():
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()


root.mainloop()

# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host="0.0.0.0", port=port)


