import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import os
from collections import deque
import time

# ==============================
# CONFIGURACIÓN
# ==============================
MODEL_PATH = "yolov8n-pose.pt"
VIDEO_PATH = r"C:\Users\Edison\Desktop\nuevo2\Avances_Proyecto\PROYECTO_CONSTRUCCION\monitoreo\data\robo.avi"

# Umbrales de velocidad de movimiento corporal
UMBRAL_MOVIMIENTO_LENTO = 8  # Movimiento normal/caminando
UMBRAL_MOVIMIENTO_RAPIDO = 25  # Movimiento sospechoso/corriendo
UMBRAL_ROBO = 40  # Movimiento muy rápido/robo en progreso

# Análisis de postura sospechosa
UMBRAL_AGACHADO = 0.35  # Ratio altura/ancho para detectar si está agachado

# Ventana temporal para análisis
VENTANA_FRAMES = 8  # Analizar últimos 8 frames

# Control de velocidad de video
TARGET_FPS = 30

# Carpeta para evidencias
ALERT_DIR = "media/alertas"
os.makedirs(ALERT_DIR, exist_ok=True)

# Cooldown entre alertas de la misma persona (segundos)
COOLDOWN_ALERTAS = 2

# ==============================
# MODELO
# ==============================
model = YOLO(MODEL_PATH)


# ==============================
# ANÁLISIS DE COMPORTAMIENTO
# ==============================
class AnalizadorComportamiento:
    def __init__(self, person_id):
        self.person_id = person_id
        self.historial_posiciones = deque(maxlen=VENTANA_FRAMES)
        self.historial_velocidades = deque(maxlen=VENTANA_FRAMES)
        self.historial_posturas = deque(maxlen=VENTANA_FRAMES)
        self.ultima_alerta = 0
        self.total_alertas = 0

    def actualizar(self, keypoints):
        """Actualiza el historial con nueva detección"""
        self.historial_posiciones.append(keypoints)

        # Calcular velocidad si hay posición previa
        if len(self.historial_posiciones) >= 2:
            velocidad = self._calcular_velocidad()
            self.historial_velocidades.append(velocidad)

        # Analizar postura
        postura = self._analizar_postura(keypoints)
        self.historial_posturas.append(postura)

    def _calcular_velocidad(self):
        """Calcula velocidad de movimiento entre frames"""
        pos_actual = self.historial_posiciones[-1]
        pos_anterior = self.historial_posiciones[-2]

        # Usar puntos clave principales (hombros, caderas)
        puntos_principales = [5, 6, 11, 12]  # Hombros y caderas

        velocidades = []
        for idx in puntos_principales:
            if idx < len(pos_actual) and idx < len(pos_anterior):
                if pos_actual[idx][0] > 0 and pos_anterior[idx][0] > 0:
                    dist = np.linalg.norm(pos_actual[idx] - pos_anterior[idx])
                    velocidades.append(dist)

        return np.mean(velocidades) if velocidades else 0

    def _analizar_postura(self, keypoints):
        """Detecta si la persona está agachada (postura sospechosa)"""
        # Obtener puntos clave
        cabeza = keypoints[0] if len(keypoints) > 0 else None
        hombros = [keypoints[5], keypoints[6]] if len(keypoints) > 6 else []
        caderas = [keypoints[11], keypoints[12]] if len(keypoints) > 12 else []

        # Validar que los puntos existan
        if cabeza is None or not hombros or not caderas:
            return "desconocido"

        validos = [p for p in [cabeza] + hombros + caderas if p[0] > 0 and p[1] > 0]
        if len(validos) < 3:
            return "desconocido"

        # Calcular dimensiones del cuerpo
        y_coords = [p[1] for p in validos]
        x_coords = [p[0] for p in validos]

        altura = max(y_coords) - min(y_coords)
        ancho = max(x_coords) - min(x_coords)

        if altura == 0:
            return "desconocido"

        ratio = altura / (ancho + 1)  # +1 para evitar división por cero

        # Determinar postura
        if ratio < UMBRAL_AGACHADO:
            return "agachado"
        else:
            return "normal"

    def detectar_comportamiento(self):
        """Analiza el comportamiento y retorna estado de alerta"""
        if len(self.historial_velocidades) < 3:
            return "INICIALIZANDO", (128, 128, 128), 0

        # Velocidad promedio reciente
        vel_promedio = np.mean(list(self.historial_velocidades)[-5:])
        vel_maxima = max(self.historial_velocidades)

        # Analizar postura predominante
        posturas_recientes = list(self.historial_posturas)[-5:]
        esta_agachado = posturas_recientes.count("agachado") >= 3

        # DETECCIÓN DE ROBO
        if vel_maxima > UMBRAL_ROBO or (vel_promedio > UMBRAL_MOVIMIENTO_RAPIDO and esta_agachado):
            return "🚨 ROBO DETECTADO", (0, 0, 255), vel_promedio

        # COMPORTAMIENTO SOSPECHOSO
        elif vel_promedio > UMBRAL_MOVIMIENTO_RAPIDO:
            return "⚠️ SOSPECHOSO", (0, 100, 255), vel_promedio

        # MOVIMIENTO RÁPIDO (corriendo normal)
        elif vel_promedio > UMBRAL_MOVIMIENTO_LENTO:
            return "🏃 MOVIMIENTO RÁPIDO", (0, 255, 255), vel_promedio

        # MOVIMIENTO NORMAL
        else:
            return "✓ NORMAL", (0, 255, 0), vel_promedio

    def debe_guardar_alerta(self, tiempo_actual):
        """Verifica si debe guardar alerta (con cooldown)"""
        if tiempo_actual - self.ultima_alerta >= COOLDOWN_ALERTAS:
            self.ultima_alerta = tiempo_actual
            self.total_alertas += 1
            return True
        return False


# ==============================
# MATCHING DE PERSONAS
# ==============================
def emparejar_personas(keypoints_actuales, analizadores, distancia_max=150):
    """Asocia detecciones actuales con analizadores existentes"""
    if not analizadores or len(keypoints_actuales) == 0:
        return {}

    # Calcular centroides de cuerpo
    centroides_actuales = []
    for kp in keypoints_actuales:
        puntos_validos = kp[kp[:, 0] > 0]
        if len(puntos_validos) > 0:
            centroides_actuales.append(np.mean(puntos_validos, axis=0))
        else:
            centroides_actuales.append(None)

    centroides_trackers = []
    for analizador in analizadores:
        if len(analizador.historial_posiciones) > 0:
            ultima_pos = analizador.historial_posiciones[-1]
            puntos_validos = ultima_pos[ultima_pos[:, 0] > 0]
            if len(puntos_validos) > 0:
                centroides_trackers.append(np.mean(puntos_validos, axis=0))
            else:
                centroides_trackers.append(None)
        else:
            centroides_trackers.append(None)

    # Emparejar por distancia mínima
    emparejamientos = {}
    usados = set()

    for i, cent_actual in enumerate(centroides_actuales):
        if cent_actual is None:
            continue

        mejor_match = None
        menor_dist = distancia_max

        for j, cent_tracker in enumerate(centroides_trackers):
            if cent_tracker is None or j in usados:
                continue

            dist = np.linalg.norm(cent_actual - cent_tracker)
            if dist < menor_dist:
                menor_dist = dist
                mejor_match = j

        if mejor_match is not None:
            emparejamientos[i] = mejor_match
            usados.add(mejor_match)

    return emparejamientos


# ==============================
# STREAM PRINCIPAL
# ==============================
def camara_seguridad_stream():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ ERROR: No se pudo abrir el video")
        return

    # Configurar velocidad de reproducción
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    if fps_original == 0:
        fps_original = 30

    delay_frame = 1.0 / TARGET_FPS

    # Sistema de tracking
    analizadores = []
    proximo_id = 0
    contador_frames = 0
    tiempo_inicio = time.time()

    print(f"🎥 Sistema de detección de robos iniciado")
    print(f"📊 Video: {fps_original:.1f} FPS → Reproducción: {TARGET_FPS} FPS")

    while True:
        inicio_loop = time.time()

        ret, frame = cap.read()
        if not ret:
            # Reiniciar video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            analizadores = []
            proximo_id = 0
            print("🔄 Video reiniciado")
            continue

        contador_frames += 1

        # Detección con YOLOv8
        results = model(frame, conf=0.4, verbose=False)
        frame_anotado = frame.copy()
        tiempo_actual = time.time()

        # Procesar detecciones
        keypoints = results[0].keypoints

        if keypoints is not None and len(keypoints.xy) > 0:
            kp_actuales = keypoints.xy.cpu().numpy()

            # Emparejar con analizadores existentes
            emparejamientos = emparejar_personas(kp_actuales, analizadores)

            # Actualizar analizadores existentes
            ids_actualizados = set()
            for idx_actual, idx_analizador in emparejamientos.items():
                analizadores[idx_analizador].actualizar(kp_actuales[idx_actual])
                ids_actualizados.add(idx_analizador)

            # Crear nuevos analizadores
            for i, kp in enumerate(kp_actuales):
                if i not in emparejamientos:
                    analizadores.append(AnalizadorComportamiento(proximo_id))
                    analizadores[-1].actualizar(kp)
                    ids_actualizados.add(len(analizadores) - 1)
                    proximo_id += 1

            # Eliminar analizadores inactivos
            analizadores = [a for idx, a in enumerate(analizadores) if idx in ids_actualizados]

            # Visualizar cada persona
            for idx_actual, idx_analizador in emparejamientos.items():
                analizador = analizadores[idx_analizador]
                kp_persona = kp_actuales[idx_actual]

                # Detectar comportamiento
                estado, color, velocidad = analizador.detectar_comportamiento()

                # Guardar alerta si es robo o sospechoso
                if "ROBO" in estado or "SOSPECHOSO" in estado:
                    if analizador.debe_guardar_alerta(tiempo_actual):
                        timestamp = datetime.datetime.now()
                        nombre_archivo = f"alerta_{estado.replace('🚨 ', '').replace('⚠️ ', '').replace(' ', '_')}_{timestamp.strftime('%Y%m%d_%H%M%S')}_ID{analizador.person_id}.jpg"
                        ruta_completa = os.path.join(ALERT_DIR, nombre_archivo)
                        cv2.imwrite(ruta_completa, frame_anotado)
                        print(f"📸 {estado} - Evidencia guardada: {nombre_archivo}")

                # Dibujar información
                puntos_validos = kp_persona[kp_persona[:, 1] > 0]
                if len(puntos_validos) > 0:
                    # Punto superior para texto
                    punto_superior = puntos_validos[np.argmin(puntos_validos[:, 1])]
                    x_texto = int(punto_superior[0])
                    y_texto = int(punto_superior[1]) - 60

                    # Panel de información
                    cv2.rectangle(frame_anotado, (x_texto - 5, y_texto - 5),
                                  (x_texto + 280, y_texto + 55), (0, 0, 0), -1)
                    cv2.rectangle(frame_anotado, (x_texto - 5, y_texto - 5),
                                  (x_texto + 280, y_texto + 55), color, 3)

                    # Textos
                    cv2.putText(frame_anotado, f"ID: {analizador.person_id}",
                                (x_texto + 5, y_texto + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)
                    cv2.putText(frame_anotado, estado,
                                (x_texto + 5, y_texto + 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, color, 2)
                    cv2.putText(frame_anotado, f"Vel: {velocidad:.1f}",
                                (x_texto + 5, y_texto + 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 255, 255), 1)

                    # Dibujar esqueleto
                    conexiones = [
                        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Brazos
                        (5, 11), (6, 12), (11, 12),  # Torso
                        (11, 13), (13, 15), (12, 14), (14, 16)  # Piernas
                    ]

                    for (pt1_idx, pt2_idx) in conexiones:
                        if pt1_idx < len(kp_persona) and pt2_idx < len(kp_persona):
                            pt1 = kp_persona[pt1_idx]
                            pt2 = kp_persona[pt2_idx]
                            if pt1[0] > 0 and pt2[0] > 0:
                                cv2.line(frame_anotado,
                                         (int(pt1[0]), int(pt1[1])),
                                         (int(pt2[0]), int(pt2[1])),
                                         color, 2)

                    # Dibujar keypoints
                    for kp in kp_persona:
                        if kp[0] > 0:
                            cv2.circle(frame_anotado, (int(kp[0]), int(kp[1])),
                                       5, color, -1)
                            cv2.circle(frame_anotado, (int(kp[0]), int(kp[1])),
                                       6, (255, 255, 255), 1)

        # Panel de estadísticas
        tiempo_transcurrido = tiempo_actual - tiempo_inicio
        fps_actual = contador_frames / tiempo_transcurrido if tiempo_transcurrido > 0 else 0
        total_alertas = sum(a.total_alertas for a in analizadores)

        # Fondo semi-transparente
        overlay = frame_anotado.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_anotado, 0.3, 0, frame_anotado)

        # Información del sistema
        cv2.putText(frame_anotado, f"FPS: {fps_actual:.1f}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_anotado, f"Personas detectadas: {len(analizadores)}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_anotado, f"Alertas guardadas: {total_alertas}",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        cv2.putText(frame_anotado, f"Frame: {contador_frames}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Codificar y enviar frame
        _, buffer = cv2.imencode('.jpg', frame_anotado,
                                 [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_bytes = buffer.tobytes()

        yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

        # Control de velocidad
        tiempo_procesamiento = time.time() - inicio_loop
        tiempo_espera = max(0, delay_frame - tiempo_procesamiento)
        time.sleep(tiempo_espera)

    cap.release()