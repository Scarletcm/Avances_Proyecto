
from django.urls import path
from . import views, api_views

app_name = 'monitoreo'

urlpatterns = [
    # RF-07: Gestión de usuarios - Autenticación
    path('', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Video stream en tiempo real
    path("video/", views.video_feed, name="video_feed"),
    path("camara/", views.video_feed2, name="video_feed2"),

    
    # Dashboard principal
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # RF-03: Gestión de alertas
    path('alertas/', views.alertas, name='alertas'),

    # Resolver una alerta
    path('alertas/<int:alerta_id>/resolver/', views.resolver_alerta, name='resolver_alerta'),

    # Ver detalle de una alerta
    path('alertas/<int:alerta_id>/', views.detalle_alerta, name='detalle_alerta'),

    # API endpoint para obtener alertas en JSON
    path('api/alertas/', views.alertas_api, name='alertas_api'),
    
    # RF-05: Registro de eventos
    path('eventos/', views.eventos_view, name='eventos'),

    path('eventos/<int:evento_id>/detalles/', views.evento_detalles_json, name='evento_detalles'),
    path('eventos/exportar/', views.exportar_eventos_csv, name='exportar_eventos'),
    path('eventos/<int:evento_id>/evidencia/', views.descargar_evidencia, name='descargar_evidencia'),

    path('eventos/<int:evento_id>/reporte/', views.generar_reporte_evento, name='reporte_evento'),

    
    # RF-06 y RF-04: Estadísticas y ubicación de incidentes
    path('estadisticas/', views.estadisticas_dashboard, name='estadisticas'),

    path("ubicacion/", views.recibir_ubicacion, name="ubicacion"),

    # API endpoints para análisis
    path('api/analyze/<int:video_id>/', api_views.analyze_video, name='api_analyze_video'),
    path('api/training-stats/', api_views.get_training_stats, name='api_training_stats'),
]
