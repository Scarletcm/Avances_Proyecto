from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("data/robo.avi")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.1, imgsz=640)

    annotated = results[0].plot()

    cv2.imshow("Test", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
