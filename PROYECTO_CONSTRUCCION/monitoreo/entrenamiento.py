# camara.py
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def camara_seguridad_stream():

    cap = cv2.VideoCapture(
        r"C:\Users\Edison\Desktop\nuevo2\Avances_Proyecto\PROYECTO_CONSTRUCCION\monitoreo\data\robo.avi"
    )

    if not cap.isOpened():
        print("ERROR: No se pudo abrir el video")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # ⭐ PASO CLAVE — detección
        results = model(frame, conf=0.2)

        # Dibujar cajas
        annotated = results[0].plot()

        # Enviar al stream
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

    cap.release()
