from ultralytics import YOLO
from models import Alertas
from .entrenamiento import camara_seguridad_stream

def alertas():
    model = YOLO("yolov8n.pt")
    results = model(frame, classes= [0])

    if results:
        Alertas.objects.create(


        )


