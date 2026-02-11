import json
import os
from django.conf import settings

def cargar_estadisticas():
    ruta_json = os.path.join(
        settings.BASE_DIR,
        'monitoreo',
        'data',
        'estadisticas.json'
    )

    with open(ruta_json, 'r', encoding='utf-8') as archivo:
        datos = json.load(archivo)

    return datos