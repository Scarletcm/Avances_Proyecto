import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import os

# ==============================
# CONFIGURACIÓN
# ==============================

MODEL_PATH = "yolov8n-pose.pt"
VIDEO_PATH =  r"C:\Users\SCARLET CASTILLO\Avances_Proyecto\PROYECTO_CONSTRUCCION\monitoreo\data\robo.avi"

UMBRAL_NORMAL = 3
UMBRAL_SOSPECHOSO = 7

# Carpeta para guardar evidencias
ALERT_DIR = "media/alertas"
os.makedirs(ALERT_DIR, exist_ok=True)

# ==============================
# MODELO
# ==============================

model = YOLO(MODEL_PATH)

# ==============================
# STREAM DE CÁMARA
# ==============================

def camara_seguridad_stream():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: No se pudo abrir el video")
        return

    prev_keypoints = []

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, conf=0.3)
        annotated = frame.copy()

        keypoints = results[0].keypoints

        if keypoints is not None:
            current = keypoints.xy.cpu().numpy()

            for i, person_kp in enumerate(current):

                # ==============================
                # ANÁLISIS DE MOVIMIENTO
                # ==============================

                tipo_movimiento = "NORMAL"
                color = (0, 255, 0)  # Verde

                if i < len(prev_keypoints):
                    diff = np.abs(person_kp - prev_keypoints[i]).mean()

                    if diff > UMBRAL_SOSPECHOSO:
                        tipo_movimiento = "SOSPECHOSO"
                        color = (0, 0, 255)  # Rojo

                        # 📸 Guardar evidencia
                        now = datetime.datetime.now()
                        filename = f"alerta_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(os.path.join(ALERT_DIR, filename), annotated)

                    elif diff > UMBRAL_NORMAL:
                        tipo_movimiento = "NORMAL"
                        color = (0, 255, 0)

                # ==============================
                # VISUALIZACIÓN
                # ==============================

                # Posición del texto (cabeza)
                x, y = int(person_kp[0][0]), int(person_kp[0][1])

                cv2.putText(
                    annotated,
                    tipo_movimiento,
                    (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )

                # Dibujar keypoints
                for kp in person_kp:
                    cv2.circle(
                        annotated,
                        (int(kp[0]), int(kp[1])),
                        3,
                        color,
                        -1
                    )

        prev_keypoints = current.copy() if keypoints is not None else prev_keypoints

        # ==============================
        # STREAM PARA DJANGO
        # ==============================

        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

    cap.release()
