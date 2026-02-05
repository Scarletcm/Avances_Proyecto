import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# =========================
# CONFIGURACIÓN
# =========================
SEQUENCE_LENGTH = 30
MODEL_PATH = "modelo_movimiento.pkl"
DATASET_X = []
DATASET_Y = []

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
# YOLOv8
# =========================
yolo_model = YOLO("yolov8n.pt")

pose_sequence = deque(maxlen=SEQUENCE_LENGTH)

# =========================
# FUNCIONES ML
# =========================
def extract_keypoints(results):
    if not results.pose_landmarks:
        return None
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(keypoints)


def flatten_sequence(sequence):
    return np.array(sequence).flatten()


def train_model():
    print("🧠 Entrenando modelo ML...")
    X = np.array(DATASET_X)
    y = np.array(DATASET_Y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=150)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"✅ Precisión del modelo: {acc:.2f}")

    joblib.dump(model, MODEL_PATH)
    print("💾 Modelo guardado")

    return model


def load_model():
    if os.path.exists(MODEL_PATH):
        print("📦 Modelo cargado")
        return joblib.load(MODEL_PATH)
    return None


def label_to_text(label):
    return {
        0: "QUIETO",
        1: "CAMINANDO",
        2: "MOVIMIENTO RAPIDO"
    }.get(label, "DESCONOCIDO")


# =========================
# STREAM DE CÁMARA
# =========================
def camara_seguridad_stream(modo="prediccion", etiqueta=0):
    """
    modo:
    - "entrenamiento": guarda datos
    - "prediccion": usa ML
    """
    cap = cv2.VideoCapture('../media/training_videos/WhatsApp_Video_2026-01-24_at_21.47.12.mp4')
    model = load_model()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_yolo = yolo_model(frame)[0]

        for box in results_yolo.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_roi = frame[y1:y2, x1:x2]
            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

            results_pose = pose_video.process(rgb_roi)
            keypoints = extract_keypoints(results_pose)

            if keypoints is not None:
                pose_sequence.append(keypoints)

            if len(pose_sequence) == SEQUENCE_LENGTH:
                features = flatten_sequence(pose_sequence)

                if modo == "entrenamiento":
                    DATASET_X.append(features)
                    DATASET_Y.append(etiqueta)
                    texto = "CAPTURANDO DATOS"

                else:
                    if model is not None:
                        pred = model.predict([features])[0]
                        texto = label_to_text(pred)
                    else:
                        texto = "SIN MODELO"

                cv2.putText(frame, texto, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if results_pose.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame[y1:y2, x1:x2],
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

    cap.release()
