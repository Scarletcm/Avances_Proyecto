from django.db import models
from django.contrib.auth.models import User

class TrainingVideo(models.Model):
    """Modelo para almacenar videos de entrenamiento"""
    
    BEHAVIOR_CHOICES = [
        ('normal', 'Comportamiento Normal'),
        ('robo', 'Intento de Robo'),
        ('agresion', 'Agresión'),
        ('sospechoso', 'Comportamiento Sospechoso'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    video = models.FileField(upload_to='training_videos/')
    behavior_type = models.CharField(max_length=20, choices=BEHAVIOR_CHOICES)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    frames_count = models.IntegerField(default=0)
    duration = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.title} - {self.behavior_type}"

class TrainedModel(models.Model):
    """Modelo entrenado guardado"""
    
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    model_file = models.FileField(upload_to='trained_models/')
    accuracy = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    training_samples = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} (Accuracy: {self.accuracy:.2%})"

class DetectionLog(models.Model):
    """Registro de detecciones realizadas"""
    
    timestamp = models.DateTimeField(auto_now_add=True)
    camera_id = models.CharField(max_length=50)
    detected_behavior = models.CharField(max_length=20, choices=TrainingVideo.BEHAVIOR_CHOICES)
    confidence = models.FloatField()
    is_alert = models.BooleanField(default=False)
    frame_data = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.camera_id} - {self.detected_behavior}"

class Ubicacion(models.Model):
     latitud = models.FloatField()
     longitud = models.FloatField()
     fecha = models.DateTimeField(auto_now_add=True)

     def __str__(self):
        return f"{self.latitud}, {self.longitud} - {self.fecha}"

