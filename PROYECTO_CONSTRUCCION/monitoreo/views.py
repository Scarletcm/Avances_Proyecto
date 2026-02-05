import json

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse
from django.utils import timezone
from django.views.decorators import gzip
from django.db.models import Count

from .models import TrainingVideo, TrainedModel, DetectionLog, Ubicacion
from .forms import LoginForm, TrainingVideoForm, TrainingBatchForm
from .services.video_service import CameraManager, VideoStreamGenerator
from .services.detection_service import detection_service, training_service
from .utils.validators import VideoValidator, TrainingValidator
from .entrenamiento import camara_seguridad_stream



import requests
from django.shortcuts import render



def mapa(request):
    ubicacion = Ubicacion.objects.latest('id')
    lat = ubicacion.latitud
    lon = ubicacion.longitud

    return render(request, 'monitoreo/mapa.html', {
        'lat': lat,
        'lon': lon,
    })




def recibir_ubicacion(request):
    if request.method == "POST":
        data = json.loads(request.body)

        Ubicacion.objects.create(
            latitud=data["lat"],
            longitud=data["lon"]
        )

        return JsonResponse({"mensaje": "Ubicación guardada"})
    return None


# ============================================================================
# SERVICIOS GLOBALES
# ============================================================================

camera_manager = CameraManager()
video_generator = VideoStreamGenerator(camera_manager)


@gzip.gzip_page
def video_feed(request):
    """
    Stream de video en tiempo real (MJPEG)
    Utiliza VideoStreamGenerator para generar frames
    """
    return StreamingHttpResponse(
        camara_seguridad_stream(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )


from pathlib import Path
import json
from django.contrib.auth.decorators import login_required
# RF-04, RF-06: ESTADÍSTICAS
# ============================================================================
@login_required(login_url='monitoreo:login')
def estadisticas(request):

    json_path = Path(__file__).resolve().parent / 'data' / 'datos.json'
    with open(json_path, encoding='utf-8') as f:
        datos = json.load(f)

    # 🔹 filtros reales desde la URL
    tipo = request.GET.get('tipo', 'all')
    zona = request.GET.get('zona', '')
    rango = request.GET.get('rango', 'month')  # (simulado)

    zonas_filtradas = {}

    for key, z in datos['zonas'].items():

        # filtro por zona
        if zona and zona not in key:
            continue

        # filtro por tipo
        if tipo == 'suspicious' and z['sospechosos'] == 0:
            continue
        if tipo == 'normal' and z['normales'] == 0:
            continue

        zonas_filtradas[key] = z

    total_eventos = sum(z['eventos'] for z in zonas_filtradas.values()) or 1

    zonas_barras = [
        {
            'zona': z['nombre'],
            'total': z['eventos'],
            'porcentaje': round((z['eventos'] / total_eventos) * 100, 1),
            'nivel': z['riesgo'].upper()
        }
        for z in zonas_filtradas.values()
    ]

    context = {
        'resumen': datos['resumen'],
        'zonas': zonas_filtradas,
        'zonas_barras': zonas_barras,
        'severidad': datos.get('severidad', {}),
        'filtro_zona': zona,
        'filtro_tipo': tipo,
        'filtro_rango': rango
    }

    return render(request, 'monitoreo/estadisticas.html', context)


# ============================================================================
# RF-07: AUTENTICACIÓN
# ============================================================================

def login_view(request):
    """
    Vista de inicio de sesión.
    Permite autenticación con usuario y contraseña.
    """
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            
            user = authenticate(request, username=username, password=password)
            
            if user is not None and user.is_active:
                login(request, user)
                return redirect('monitoreo:dashboard')
            else:
                form.add_error(None, 'Usuario o contraseña incorrectos')
    else:
        form = LoginForm()
    
    return render(request, 'monitoreo/login.html', {'form': form})


def logout_view(request):
    """Vista para cerrar sesión"""
    logout(request)
    return redirect('monitoreo:login')


# ============================================================================
# RF-01, RF-02, RF-03: DASHBOARD
# ============================================================================

@login_required(login_url='monitoreo:login')
def dashboard(request):
    """
    Dashboard principal del sistema.
    
    RF-01: Análisis de video en tiempo real
    RF-02: Detección de comportamientos
    RF-03: Generación de alertas automáticas
    """
    context = {
        'page_title': 'Dashboard de Monitoreo',
        'user': request.user,
        'model_active': detection_service.is_model_trained(),
        'model_info': detection_service.get_active_model_info(),
    }
    return render(request, 'monitoreo/dashboard.html', context)


# ============================================================================
# RF-03: ALERTAS
# ============================================================================

@login_required(login_url='monitoreo:login')
def alertas(request):
    """
    Panel de alertas.
    RF-03: Generar alertas automáticas
    """
    context = {
        'page_title': 'Panel de Alertas',
        'user': request.user,
    }
    return render(request, 'monitoreo/alertas.html', context)


# ============================================================================
# RF-05: EVENTOS
# ============================================================================

@login_required(login_url='monitoreo:login')
def eventos(request):
    """
    Registro de eventos detectados.
    RF-05: Registrar eventos detectados
    """
    context = {
        'page_title': 'Registro de Eventos',
        'user': request.user,
    }
    return render(request, 'monitoreo/eventos.html', context)


# ============================================================================
# RF-04, RF-06: ESTADÍSTICAS
# ============================================================================




# ============================================================================
# VISTAS LEGACY (Compatibilidad)
# ============================================================================

@login_required(login_url='monitoreo:login')
def training_videos(request):
    """Vista legacy - redirecciona al centro de entrenamiento"""
    return redirect('monitoreo:training_center')
