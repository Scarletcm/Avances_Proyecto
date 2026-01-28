"""
Detection Service - Lógica de detección y predicción
Wrapper alrededor de BehaviorDetector
"""

import os
import shutil
from django.conf import settings
from ..behavior_detector import detector
from ..models import TrainingVideo, TrainedModel


class DetectionService:
    """Servicio centralizado para detección de comportamientos"""
    
    def __init__(self):
        self.detector = detector
    
    def is_model_trained(self):
        """Verifica si hay un modelo entrenado activo"""
        return TrainedModel.objects.filter(is_active=True).exists() and self.detector.is_trained
    
    def get_active_model_info(self):
        """Obtiene información del modelo activo"""
        model = TrainedModel.objects.filter(is_active=True).first()
        
        if not model:
            return None
        
        return {
            'name': model.name,
            'accuracy': model.accuracy,
            'precision': model.precision,
            'recall': model.recall,
            'f1_score': model.f1_score,
            'samples': model.training_samples,
            'created_at': model.created_at
        }
    
    def predict_frame(self, frame):
        """Predice comportamiento en un frame"""
        if not self.is_model_trained():
            return None, 0.0
        
        prediction, confidence = self.detector.predict_frame(frame)
        
        if prediction is None:
            return None, 0.0
        
        behavior = self.detector.reverse_map.get(prediction, 'desconocido')
        return behavior, float(confidence)
    
    def get_behavior_labels(self):
        """Obtiene lista de comportamientos detectables"""
        return list(self.detector.label_map.keys())


class TrainingService:
    """Servicio para entrenar modelos"""
    
    def __init__(self):
        self.detector = detector
    
    def prepare_training_data(self):
        """Prepara estructura de datos para entrenamiento"""
        base_path = os.path.join(settings.MEDIA_ROOT, 'training_structure')
        
        # Crear estructura de carpetas
        os.makedirs(base_path, exist_ok=True)
        
        for behavior in ['normal', 'robo', 'agresion', 'sospechoso']:
            behavior_dir = os.path.join(base_path, behavior)
            os.makedirs(behavior_dir, exist_ok=True)
        
        # Copiar videos no procesados a estructura
        videos = TrainingVideo.objects.filter(processed=False)
        
        for video in videos:
            if video.video:
                src = video.video.path
                
                if os.path.exists(src):
                    dst_dir = os.path.join(base_path, video.behavior_type)
                    dst = os.path.join(dst_dir, os.path.basename(video.video.name))
                    
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"Error copiando {src}: {str(e)}")
        
        return base_path
    
    def get_training_stats(self):
        """Obtiene estadísticas de videos de entrenamiento"""
        videos = TrainingVideo.objects.all()
        
        return {
            'total': videos.count(),
            'normal': videos.filter(behavior_type='normal').count(),
            'sospechoso': videos.filter(behavior_type='sospechoso').count(),
            'agresion': videos.filter(behavior_type='agresion').count(),
            'robo': videos.filter(behavior_type='robo').count(),
            'processed': videos.filter(processed=True).count(),
            'unprocessed': videos.filter(processed=False).count(),
        }
    
    def train_model(self, test_size=0.2):
        """Entrena un nuevo modelo con datos disponibles"""
        
        # Preparar datos
        base_path = self.prepare_training_data()
        
        # Entrenar
        metrics = self.detector.train(base_path, test_size=test_size)
        
        # Guardar en base de datos
        model = TrainedModel.objects.create(
            name=f"Model - {TrainedModel.objects.count() + 1}",
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1'],
            training_samples=metrics['samples'],
            is_active=True
        )
        
        # Marcar videos como procesados
        TrainingVideo.objects.filter(processed=False).update(processed=True)
        
        # Desactivar modelos anteriores
        TrainedModel.objects.exclude(id=model.id).update(is_active=False)
        
        return {
            'model_id': model.id,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1'],
            'samples': metrics['samples']
        }
    
    def validate_training_data(self):
        """Valida que haya suficientes datos para entrenar"""
        stats = self.get_training_stats()
        
        min_per_type = 5
        
        issues = []
        for behavior in ['normal', 'sospechoso', 'agresion', 'robo']:
            count = stats[behavior]
            if count < min_per_type:
                issues.append(
                    f"{behavior}: {count} videos (mínimo {min_per_type} recomendado)"
                )
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }


# Instancias globales
detection_service = DetectionService()
training_service = TrainingService()
