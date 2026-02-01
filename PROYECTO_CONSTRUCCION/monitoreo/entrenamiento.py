import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# =========================
# MediaPipe
# =========================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose_image = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)

pose_video = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Dataset (Machine Learning)
# =========================
X = []
y = []

LABEL_POSE_CORRECTA = 0
LABEL_POSE_INCORRECTA = 1

labels_map = {
    0: "POSE CORRECTA",
    1: "POSE INCORRECTA"
}

# =========================
# Modelo ML
# =========================
model = KNeighborsClassifier(n_neighbors=3)
trained = False

# =========================
# Funciones ML
# =========================
def extract_keypoints(results):
    if not results.pose_landmarks:
        return None

    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(keypoints)


def load_reference_pose(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Imagen no encontrada")
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_image.process(rgb)

    if not results.pose_landmarks:
        print("❌ No se detectó pose en la imagen")
        return None

    print("✅ Pose referencia cargada")
    return extract_keypoints(results)


def train_model():
    global trained
    if len(X) >= 3:
        model.fit(X, y)
        trained = True
        print("🔥 MODELO ENTRENADO")


def predict_pose(keypoints):
    pred = model.predict([keypoints])[0]
    prob = model.predict_proba([keypoints])[0]
    confidence = np.max(prob)
    return pred, confidence


# =========================
# Cargar POSE REFERENCIA
# =========================
ref_pose = load_reference_pose("data/img.png")

if ref_pose is not None:
    X.append(ref_pose)
    y.append(LABEL_POSE_CORRECTA)
    train_model()


# =========================
# STREAM DE VIDEO (WEB)
# =========================
def inicio_camara1():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_video.process(rgb)

        keypoints = extract_keypoints(results)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # ========= MACHINE LEARNING =========
        if trained and keypoints is not None:
            pred, conf = predict_pose(keypoints)

            if conf > 0.8:
                text = f"{labels_map[pred]} ({conf:.2f})"
                color = (0, 255, 0)
            elif conf > 0.6:
                text = f"CASI IGUAL ({conf:.2f})"
                color = (0, 255, 255)
            else:
                text = f"NO COINCIDE ({conf:.2f})"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        # ========= SALIDA PARA HTML =========
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

    cap.release()
