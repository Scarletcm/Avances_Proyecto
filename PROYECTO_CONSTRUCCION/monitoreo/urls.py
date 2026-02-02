
from django.urls import path
from . import views, api_views

app_name = 'monitoreo'

urlpatterns = [
    # RF-07: Gestión de usuarios - Autenticación
    path('', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Video stream en tiempo real
    path('video_feed/', views.video_feed, name='video_feed'),
    
    # Dashboard principal
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # RF-03: Gestión de alertas
    path('alertas/', views.alertas, name='alertas'),
    
    # RF-05: Registro de eventos
    path('eventos/', views.eventos, name='eventos'),
    
    # RF-06 y RF-04: Estadísticas y ubicación de incidentes
    path('estadisticas/', views.estadisticas, name='estadisticas'),

    path("ubicacion/", views.recibir_ubicacion, name="ubicacion"),

    # API endpoints para análisis
    path('api/analyze/<int:video_id>/', api_views.analyze_video, name='api_analyze_video'),
    path('api/training-stats/', api_views.get_training_stats, name='api_training_stats'),
]
