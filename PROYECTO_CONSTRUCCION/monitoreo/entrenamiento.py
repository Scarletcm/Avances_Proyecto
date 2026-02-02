import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from collections import deque
from django.shortcuts import render

# =========================
# MediaPipe Pose
# =========================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose_video = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# YOLOv8 Detector de personas
# =========================
yolo_model = YOLO("yolov8n.pt")  # modelo ligero

# =========================
# Secuencia de poses
# =========================
sequence_length = 30
pose_sequence = deque(maxlen=sequence_length)


# =========================
# Funciones
# =========================
def extract_keypoints(results):
    if not results.pose_landmarks:
        return None
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(keypoints)


def analyze_sequence(sequence):
    """Ejemplo simple de patrón de movimiento"""
    if len(sequence) < 2:
        return "Esperando secuencia..."

    diffs = [np.linalg.norm(sequence[i] - sequence[i - 1]) for i in range(1, len(sequence))]
    avg_speed = np.mean(diffs)

    if avg_speed > 0.5:
        return "MOVIMIENTO RÁPIDO"
    elif avg_speed > 0.1:
        return "MOVIMIENTO NORMAL"
    else:
        return "POSE ESTÁTICA"


# =========================
# Stream de video para HTML
# =========================
def camara_seguridad_stream():
    cap = cv2.VideoCapture(0)  # Cambia 0 por tu IP RTSP si es remoto

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # =========================
        # Detectar personas
        # =========================
        results_yolo = yolo_model(frame)[0]
        for box in results_yolo.boxes:
            cls = int(box.cls[0])
            if cls != 0:  # 0 = persona
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_roi = frame[y1:y2, x1:x2]
            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

            # =========================
            # Detectar pose
            # =========================
            results_pose = pose_video.process(rgb_roi)
            keypoints = extract_keypoints(results_pose)

            if keypoints is not None:
                pose_sequence.append(keypoints)

            # Dibujar bounding box y pose
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if results_pose.pose_landmarks:
                mp_draw.draw_landmarks(frame[y1:y2, x1:x2], results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # =========================
        # Analizar secuencia
        # =========================
        pattern_text = analyze_sequence(list(pose_sequence))
        cv2.putText(frame, pattern_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # =========================
        # Codificar frame para HTML
        # =========================
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

    cap.release()
