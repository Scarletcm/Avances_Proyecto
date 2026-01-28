"""
Vistas AJAX para captura en tiempo real y análisis
"""

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.utils import timezone
import cv2
import json
import os
from .models import TrainingVideo
from .behavior_detector import detector


@login_required(login_url='monitoreo:login')
def capture_training_video(request):
    """
    Captura video de entrenamiento desde la cámara en tiempo real.
    Aún no implementado - para futuro desarrollo
    """
    context = {
        'page_title': 'Capturar Video de Entrenamiento',
        'user': request.user,
    }
    from django.shortcuts import render
    return render(request, 'monitoreo/capture_training.html', context)


@login_required(login_url='monitoreo:login')
@require_POST
@csrf_exempt
def analyze_video(request):
    """
    Analiza un video subido para obtener predicciones
    """
    try:
        video_id = request.POST.get('video_id')
        video = TrainingVideo.objects.get(id=video_id)
        
        if not detector.is_trained:
            return JsonResponse({
                'success': False,
                'error': 'Modelo no entrenado aún'
            })
        
        # Analizar video
        detections = detector.detect_in_video(video.video.path)
        
        # Calcular estadísticas
        total_frames = len(detections)
        behavior_counts = {}
        for detection in detections:
            behavior = detection['behavior']
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        
        # Calcular confianza promedio
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0
        
        # Comportamiento predominante
        if behavior_counts:
            predominant_behavior = max(behavior_counts.items(), key=lambda x: x[1])[0]
        else:
            predominant_behavior = "Desconocido"
        
        return JsonResponse({
            'success': True,
            'total_frames': total_frames,
            'behavior_counts': behavior_counts,
            'predominant_behavior': predominant_behavior,
            'average_confidence': round(avg_confidence, 3),
            'sample_detections': detections[:10]  # Primeras 10 detecciones
        })
    except TrainingVideo.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Video no encontrado'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required(login_url='monitoreo:login')
def get_training_stats(request):
    """
    Obtiene estadísticas de entrenamiento en tiempo real
    """
    from .models import TrainingVideo, TrainedModel
    from django.db.models import Count
    
    videos = TrainingVideo.objects.values('behavior_type').annotate(count=Count('id'))
    models = TrainedModel.objects.filter(is_active=True).first()
    
    stats = {
        'videos_by_type': {item['behavior_type']: item['count'] for item in videos},
        'total_videos': TrainingVideo.objects.count(),
        'model_info': {
            'is_active': bool(models),
            'accuracy': models.accuracy if models else 0,
            'f1_score': models.f1_score if models else 0,
            'created_at': models.created_at.isoformat() if models else None
        }
    }
    
    return JsonResponse(stats)
