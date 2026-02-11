# camara.py
import cv2
import numpy as np
from ultralytics import YOLO

# ⭐ Modelo de pose corporal
model = YOLO("yolov8n-pose.pt")

def camara_seguridad_stream():

    cap = cv2.VideoCapture(
        r"C:\Users\maxxi\OneDrive\Escritorio\Proyecto\Avances_Proyecto\PROYECTO_CONSTRUCCION\monitoreo\data\robo.avi"
    )

    if not cap.isOpened():
        print("ERROR: No se pudo abrir el video")
        return

    prev_keypoints = None

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # ⭐ Detección de pose
        results = model(frame, conf=0.3)

        annotated = results[0].plot()

        # ⭐ Extraer keypoints
        keypoints = results[0].keypoints

        movimiento_detectado = False

        if keypoints is not None:
            current = keypoints.xy.cpu().numpy()

            if prev_keypoints is not None:

                # Ajustar tamaño si cambia número de personas
                min_len = min(len(current), len(prev_keypoints))

                if min_len > 0:
                    diff = np.abs(current[:min_len] - prev_keypoints[:min_len]).mean()

                    # 🔥 Umbral de movimiento (ajustable)
                    if diff > 4:
                        movimiento_detectado = True

            prev_keypoints = current

        # ⭐ Mostrar alerta visual
        if movimiento_detectado:
            cv2.putText(
                annotated,
                "MOVIMIENTO CORPORAL DETECTADO",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        # ⭐ Enviar al stream
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

    cap.release()
