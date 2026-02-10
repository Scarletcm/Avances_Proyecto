"""
Camera Service - Gestión centralizada de cámaras y video streaming
"""

import cv2
import threading
import time
from django.utils import timezone
from .optical_flow_service import OpticalFlowService



class CameraManager:
    """Singleton para gestionar la instancia de cámara"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.camera = None
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self._initialized = True
    
    def get_camera(self, camera_id=0, width=1280, height=720, fps=30):
        """Obtiene instancia de cámara con configuración específica"""
        if self.camera is None:
            self.camera = cv2.VideoCapture(camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)
        return self.camera
    
    def capture_frame(self, camera_id=0):
        """Captura un frame de la cámara"""
        camera = self.get_camera(camera_id)
        ret, frame = camera.read()
        
        if ret:
            with self.frame_lock:
                self.current_frame = frame
            return frame
        
        return None
    
    def add_metadata(self, frame, camera_name="CAM-05", location="Avenida Principal"):
        """Agrega metadata visual al frame"""
        if frame is None:
            return None
        
        frame_copy = frame.copy()
        
        # Nombre de cámara y ubicación
        cv2.putText(
            frame_copy,
            f"{camera_name} | {location}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Timestamp
        timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(
            frame_copy,
            f"Timestamp: {timestamp}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        return frame_copy
    
    def release(self):
        """Libera los recursos de la cámara"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def __del__(self):
        self.release()


class VideoStreamGenerator:
    """Generador de frames para streaming MJPEG + Optical Flow"""

    def __init__(self, camera_manager=None, frame_quality=95):
        self.camera_manager = camera_manager or CameraManager()
        self.frame_quality = frame_quality
        self.fps = 30
        self.frame_delay = 1.0 / self.fps
        self.optical_flow = OpticalFlowService()

    def generate_frames(self):
        while True:
            try:
                frame = self.camera_manager.capture_frame()

                if frame is None:
                    print("⚠️ No se pudo capturar frame")
                    continue

                # 🔥 OPTICAL FLOW
                motion_data = self.optical_flow.process(frame)

                if motion_data and motion_data["motion_level"] > 1.5:
                    cv2.putText(
                        frame,
                        f"Movimiento: {motion_data['motion_level']:.2f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                frame = self.camera_manager.add_metadata(frame)

                ret, buffer = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.frame_quality]
                )

                if not ret:
                    continue

                yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + buffer.tobytes() + b'\r\n'
                )

                time.sleep(self.frame_delay)

            except Exception as e:
                print("🔥 ERROR STREAM:", e)
