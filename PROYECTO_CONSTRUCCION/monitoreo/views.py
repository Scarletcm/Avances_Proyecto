import csv
import datetime

from django.contrib import messages
from django.core.paginator import Paginator
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse
from django.views.decorators import gzip
from sympy import Q
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.core.paginator import Paginator
from django.db.models import Q, Count
import csv
from datetime import datetime, timedelta
from .models import TrainingVideo, TrainedModel, DetectionLog, Ubicacion, Alertas
from .forms import LoginForm, TrainingVideoForm, TrainingBatchForm
from .services.video_service import CameraManager, VideoStreamGenerator
from .services.detection_service import detection_service, training_service
from .entrenamiento import camara_seguridad_stream
import requests
from django.shortcuts import render
from django.utils import timezone


# MAPA
def mapa(request):
    ubicacion = Ubicacion.objects.latest('id')

    return render(request, 'monitoreo/mapa.html', {
        'lat': ubicacion.latitud,
        'lon': ubicacion.longitud,
        'ciudad': ubicacion.ciudad
    })


# RECIBIR UBICACION
def recibir_ubicacion(request):

    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    # ---------- Leer JSON ----------
    try:
        data = json.loads(request.body.decode("utf-8"))
        lat = float(data["lat"])
        lon = float(data["lon"])
    except json.JSONDecodeError:
        return JsonResponse({"error": "JSON inválido"}, status=400)
    except (KeyError, ValueError):
        return JsonResponse({"error": "Datos lat/lon inválidos"}, status=400)

    # ---------- Obtener ciudad ----------
    ciudad = "Desconocida"

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "monitoreo_app"},
            timeout=8
        )

        if response.status_code == 200:
            info = response.json()
            addr = info.get("address", {})

            keys = [
                "city",
                "town",
                "municipality",
                "county",
                "state_district",
                "region",
                "state"
            ]

            for k in keys:
                if k in addr:
                    ciudad = addr[k]
                    break

    except requests.RequestException:
        # NO rompemos el sistema si falla internet
        pass

    # ---------- Guardar SIEMPRE nueva ubicación ----------
    ubicacion = Ubicacion.objects.create(
        latitud=lat,
        longitud=lon,
        ciudad=ciudad
    )

    # ---------- Crear alerta ----------
    Alertas.objects.create(
        ubicacion=ubicacion,
        comportamiento="Movimiento Sospechoso",
        severidad="Alta",
        hora=timezone.now(),
        descripcion="Movimiento detectado",
        estado="Activo"
    )

    return JsonResponse({
        "mensaje": "Ubicación guardada",
        "ciudad": ciudad
    })

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
    Panel de alertas con filtros y estadísticas.
    RF-03: Generar alertas automáticas
    """
    # Obtener todas las alertas ordenadas por hora
    alertas_queryset = Alertas.objects.all().order_by("-hora")

    # Aplicar filtros si existen
    severidad_filter = request.GET.get('severidad', '')
    estado_filter = request.GET.get('estado', '')

    if severidad_filter:
        alertas_queryset = alertas_queryset.filter(severidad=severidad_filter)

    if estado_filter:
        alertas_queryset = alertas_queryset.filter(estado=estado_filter)

    # Calcular estadísticas
    estadisticas = {
        'total': alertas_queryset.count(),
        'alta': alertas_queryset.filter(severidad='Alta').count(),
        'media': alertas_queryset.filter(severidad='Media').count(),
        'baja': alertas_queryset.filter(severidad='Baja').count(),
        'pendientes': alertas_queryset.filter(estado='Pendiente').count(),
        'activas': alertas_queryset.filter(estado='Activo').count(),
    }

    context = {
        'alerta': alertas_queryset,
        'estadisticas': estadisticas,
        'severidad_filter': severidad_filter,
        'estado_filter': estado_filter,
    }

    return render(request, 'monitoreo/alertas.html', context)


def resolver_alerta(request, alerta_id):
    """
    Marca una alerta como resuelta (cambia el estado).
    """
    if request.method == 'POST':
        alerta = get_object_or_404(Alertas, id=alerta_id)

        # Cambiar el estado a Activo (o podrías crear un nuevo estado "Resuelta")
        alerta.estado = 'Activo'
        alerta.save()

        messages.success(request, f'Alerta #{alerta_id} resuelta correctamente.')

    return redirect('alertas')


def detalle_alerta(request, alerta_id):
    """
    Muestra el detalle completo de una alerta específica.
    """
    alerta = get_object_or_404(Alertas, id=alerta_id)

    context = {
        'alerta': alerta,
    }

    return render(request, 'monitoreo/detalle_alerta.html', context)


def alertas_api(request):
    """
    API endpoint para obtener alertas en formato JSON.
    Útil para actualizaciones en tiempo real con AJAX.
    """
    alertas_queryset = Alertas.objects.all().order_by("-hora")

    # Filtros opcionales
    severidad = request.GET.get('severidad', '')
    estado = request.GET.get('estado', '')

    if severidad:
        alertas_queryset = alertas_queryset.filter(severidad=severidad)

    if estado:
        alertas_queryset = alertas_queryset.filter(estado=estado)

    # Serializar datos
    alertas_data = []
    for alerta in alertas_queryset:
        alertas_data.append({
            'id': alerta.id,
            'ubicacion': alerta.ubicacion.nombre,
            'comportamiento': alerta.comportamiento,
            'severidad': alerta.severidad,
            'hora': alerta.hora.strftime('%Y-%m-%d %H:%M:%S'),
            'descripcion': alerta.descripcion,
            'estado': alerta.estado,
        })

    return JsonResponse({
        'alertas': alertas_data,
        'total': len(alertas_data)
    })
# ============================================================================
# RF-05: EVENTOS
# ============================================================================

@login_required(login_url='monitoreo:login')
def eventos_view(request):

    # Obtener todas las alertas
    alertas = Alertas.objects.select_related('ubicacion').all().order_by('-hora')

    # Filtros
    search = request.GET.get('search', '')
    severidad = request.GET.get('severidad', '')
    estado = request.GET.get('estado', '')
    ubicacion_id = request.GET.get('ubicacion', '')

    # Aplicar filtros
    if search:
        alertas = alertas.filter(
            Q(comportamiento__icontains=search) |
            Q(descripcion__icontains=search) |
            Q(ubicacion__nombre__icontains=search)
        )

    if severidad:
        alertas = alertas.filter(severidad=severidad)

    if estado:
        alertas = alertas.filter(estado=estado)

    if ubicacion_id:
        alertas = alertas.filter(ubicacion_id=ubicacion_id)

    # Estadísticas
    total_eventos = alertas.count()
    eventos_alta = alertas.filter(severidad='Alta').count()
    eventos_media = alertas.filter(severidad='Media').count()
    eventos_baja = alertas.filter(severidad='Baja').count()

    # Paginación
    paginator = Paginator(alertas, 10)  # 10 eventos por página
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    # Obtener todas las ubicaciones para el filtro
    ubicaciones = Ubicacion.objects.all()

    context = {
        'alertas': page_obj,
        'page_obj': page_obj,
        'is_paginated': page_obj.has_other_pages(),
        'total_eventos': total_eventos,
        'eventos_alta': eventos_alta,
        'eventos_media': eventos_media,
        'eventos_baja': eventos_baja,
        'ubicaciones': ubicaciones,
    }

    return render(request, 'monitoreo/eventos.html', context)


def evento_detalles_json(request, evento_id):
    """
    API endpoint para obtener detalles de un evento en formato JSON
    Utilizado por el modal de detalles
    """
    alerta = get_object_or_404(Alertas, id=evento_id)

    data = {
        'id': alerta.id,
        'hora': alerta.hora.strftime('%Y-%m-%d %H:%M:%S'),
        'ubicacion': f"{alerta.ubicacion.nombre} - {alerta.ubicacion.zona}",
        'comportamiento': alerta.comportamiento,
        'severidad': alerta.severidad,
        'estado': alerta.estado,
        'descripcion': alerta.descripcion,
    }

    return JsonResponse(data)


def exportar_eventos_csv(request):
    """
    Exportar todos los eventos a formato CSV
    """
    # Obtener eventos con filtros si se aplicaron
    alertas = Alertas.objects.select_related('ubicacion').all().order_by('-hora')

    # Aplicar mismos filtros que en la vista principal
    search = request.GET.get('search', '')
    severidad = request.GET.get('severidad', '')
    estado = request.GET.get('estado', '')
    ubicacion_id = request.GET.get('ubicacion', '')

    if search:
        alertas = alertas.filter(
            Q(comportamiento__icontains=search) |
            Q(descripcion__icontains=search) |
            Q(ubicacion__nombre__icontains=search)
        )

    if severidad:
        alertas = alertas.filter(severidad=severidad)

    if estado:
        alertas = alertas.filter(estado=estado)

    if ubicacion_id:
        alertas = alertas.filter(ubicacion_id=ubicacion_id)

    # Crear respuesta CSV
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="eventos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
    response.write('\ufeff')  # BOM para Excel

    writer = csv.writer(response)
    writer.writerow([
        'ID',
        'Fecha/Hora',
        'Ubicación',
        'Zona',
        'Comportamiento',
        'Severidad',
        'Estado',
        'Descripción'
    ])

    for alerta in alertas:
        writer.writerow([
            alerta.id,
            alerta.hora.strftime('%Y-%m-%d %H:%M:%S'),
            alerta.ubicacion.nombre,
            alerta.ubicacion.zona,
            alerta.comportamiento,
            alerta.severidad,
            alerta.estado,
            alerta.descripcion
        ])

    return response


def descargar_evidencia(request, evento_id):
    """
    Descargar evidencia relacionada con un evento específico
    """
    alerta = get_object_or_404(Alertas, id=evento_id)

    # Aquí puedes implementar la lógica para descargar imágenes, videos, etc.
    # Por ahora, retorna un archivo de texto con los detalles

    response = HttpResponse(content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="evidencia_evento_{alerta.id}.txt"'

    content = f"""
REPORTE DE EVIDENCIA - EVENTO #{alerta.id}
{'=' * 50}

Fecha y Hora: {alerta.hora.strftime('%Y-%m-%d %H:%M:%S')}
Ubicación: {alerta.ubicacion.nombre}
Zona: {alerta.ubicacion.zona}
Comportamiento Detectado: {alerta.comportamiento}
Severidad: {alerta.severidad}
Estado: {alerta.estado}

Descripción:
{alerta.descripcion}

{'=' * 50}
Este documento fue generado automáticamente por el Sistema de Monitoreo.
Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    response.write(content)
    return response


def generar_reporte_evento(request, evento_id):

    alerta = get_object_or_404(Alertas, id=evento_id)

    response = HttpResponse(content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="reporte_evento_{alerta.id}.txt"'

    content = f"""
{'=' * 60}
        REPORTE DETALLADO DE EVENTO DE SEGURIDAD
{'=' * 60}

INFORMACIÓN GENERAL
-------------------
ID del Evento: #{alerta.id}
Fecha: {alerta.hora.strftime('%d/%m/%Y')}
Hora: {alerta.hora.strftime('%H:%M:%S')}
Estado Actual: {alerta.estado}

UBICACIÓN
---------
Nombre: {alerta.ubicacion.nombre}
Zona: {alerta.ubicacion.zona}
Coordenadas: {alerta.ubicacion.latitud}, {alerta.ubicacion.longitud}

DETALLES DEL EVENTO
-------------------
Tipo de Comportamiento: {alerta.comportamiento}
Nivel de Severidad: {alerta.severidad}

DESCRIPCIÓN COMPLETA
--------------------
{alerta.descripcion if alerta.descripcion else 'Sin descripción adicional'}

RECOMENDACIONES
---------------
"""

    # Recomendaciones basadas en severidad
    if alerta.severidad == 'Alta':
        content += """
- Requiere atención INMEDIATA del personal de seguridad
- Verificar y activar protocolos de emergencia si es necesario
- Documentar todas las acciones tomadas
- Notificar a supervisores y autoridades competentes
"""
    elif alerta.severidad == 'Media':
        content += """
- Requiere seguimiento por parte del personal de seguridad
- Verificar situación en campo
- Mantener monitoreo constante del área
- Documentar evolución del evento
"""
    else:
        content += """
- Mantener vigilancia rutinaria
- Registrar en bitácora estándar
- No requiere acción inmediata
"""

    content += f"""

{'=' * 60}
Reporte generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Sistema de Monitoreo y Detección de Comportamiento Sospechoso
{'=' * 60}
    """

    response.write(content)
    return response
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
