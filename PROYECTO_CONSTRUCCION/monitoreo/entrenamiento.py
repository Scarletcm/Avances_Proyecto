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
X = []  # vectores de pose (132)
y = []  # etiquetas

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
# Funciones
# =========================
def extract_keypoints(results):
    """Convierte landmarks en vector ML"""
    if not results.pose_landmarks:
        return None

    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(keypoints)

def load_reference_pose(image_path):
    """Carga pose referencia desde imagen"""
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

    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("Pose Referencia", img)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()

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
# Webcam
# =========================
cap = cv2.VideoCapture(0)

print("""
INSTRUCCIONES:
S = Guardar pose correcta
I = Guardar pose incorrecta
ESC = Salir
""")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_video.process(rgb)

    keypoints = extract_keypoints(results)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    key = cv2.waitKey(1) & 0xFF

    # Guardar ejemplos
    if key == ord('s') and keypoints is not None:
        X.append(keypoints)
        y.append(LABEL_POSE_CORRECTA)
        train_model()
        print("✔ Pose correcta guardada")

    if key == ord('i') and keypoints is not None:
        X.append(keypoints)
        y.append(LABEL_POSE_INCORRECTA)
        train_model()
        print("✔ Pose incorrecta guardada")

    # Predicción
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
            frame, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

    cv2.imshow("Comparacion de Pose (IA)", frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()