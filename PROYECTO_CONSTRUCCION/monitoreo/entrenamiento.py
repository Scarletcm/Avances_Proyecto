# camara.py
import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import os

# ==================================================
# CONFIGURACIÓN GENERAL (AJUSTABLE)
# ==================================================

MODEL_PATH = "yolov8n-pose.pt"
VIDEO_PATH = r"C:\Users\SCARLET CASTILLO\Avances_Proyecto\PROYECTO_CONSTRUCCION\monitoreo\data\robo.avi"

# Umbrales ajustados para forcejeo
UMBRAL_CUERPO = 4
UMBRAL_BRAZOS = 3
FRAMES_SOSPECHOSOS = 4  # frames consecutivos

ALERT_DIR = "media/alertas"
os.makedirs(ALERT_DIR, exist_ok=True)

# ==================================================
# MODELO
# ==================================================

model = YOLO(MODEL_PATH)

# ==================================================
# FUNCIÓN PARA DETECTAR FORCEJEO (BRAZOS)
# ==================================================

def movimiento_brazos(curr, prev):
    # Hombros, codos y muñecas (YOLOv8 Pose)
    idx = [5, 6, 7, 8, 9, 10]
    return np.abs(curr[idx] - prev[idx]).mean()

# ==================================================
# STREAM DE CÁMARA
# ==================================================

def camara_seguridad_stream():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: No se pudo abrir el video")
        return

    prev_keypoints = []
    contador_sospecha = []

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

            # Asegurar tamaño del contador
            while len(contador_sospecha) < len(current):
                contador_sospecha.append(0)

            for i, curr in enumerate(current):

                tipo = "NORMAL"
                color = (0, 255, 0)

                if i < len(prev_keypoints):
                    prev = prev_keypoints[i]

                    # Movimiento general del cuerpo
                    diff_cuerpo = np.abs(curr - prev).mean()

                    # Movimiento violento de brazos (forcejeo)
                    diff_brazos = movimiento_brazos(curr, prev)

                    # Acumulación temporal
                    if diff_cuerpo > UMBRAL_CUERPO or diff_brazos > UMBRAL_BRAZOS:
                        contador_sospecha[i] += 1
                    else:
                        contador_sospecha[i] = max(0, contador_sospecha[i] - 1)

                    # Confirmación de sospecha
                    if contador_sospecha[i] >= FRAMES_SOSPECHOSOS:
                        tipo = "SOSPECHOSO"
                        color = (0, 0, 255)

                        # Guardar evidencia
                        now = datetime.datetime.now()
                        filename = f"alerta_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(os.path.join(ALERT_DIR, filename), annotated)

                # Posición del texto (cabeza)
                x, y = int(curr[0][0]), int(curr[0][1])

                cv2.putText(
                    annotated,
                    tipo,
                    (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )

                # Dibujar keypoints
                for kp in curr:
                    cv2.circle(
                        annotated,
                        (int(kp[0]), int(kp[1])),
                        3,
                        color,
                        -1
                    )

        prev_keypoints = current.copy() if keypoints is not None else prev_keypoints

        # ==================================================
        # STREAM PARA DJANGO
        # ==================================================

        _, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

    cap.release()
