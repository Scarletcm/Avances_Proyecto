# Sistema de Seguridad Inteligente con Detección de Comportamientos Sospechosos

## 📌 Descripción del Proyecto
Este proyecto es un **Sistema de Seguridad Inteligente** desarrollado con **Django**, **OpenCV** y **YOLOv8**, cuyo objetivo es analizar videos de cámaras de seguridad para **detectar comportamientos normales y sospechosos** de personas en tiempo real o en video grabado.

El sistema se enfoca principalmente en:
- Detectar personas en video
- Analizar la **velocidad y patrones de movimiento**
- Clasificar el comportamiento como:
  - 🟢 **Normal** (movimientos lentos o cotidianos)
  - 🔴 **Sospechoso** (movimientos rápidos, forcejeos, robos)
- Generar **alertas automáticas** cuando se detecta un evento sospechoso

Este proyecto está orientado a escenarios como **robos, agresiones o forcejeos**, apoyando la toma de decisiones en sistemas de videovigilancia.

---

## 🧠 Tecnologías Utilizadas

- **Python 3**
- **Django** – Backend y gestión del sistema
- **OpenCV** – Procesamiento de video e imágenes
- **YOLOv8** – Detección de personas
- **PyTorch** – Carga de modelos entrenados (.pt)
- **SQLite3** – Base de datos

---

## 📂 Estructura del Proyecto (Resumen)

```
PROYECTO_CONSTRUCCION/
│
├── media/
│   └── training_structure/
│       ├── agresion/
│       ├── normal/
│       └── robo/
│
├── monitoreo/
│   ├── services/
│   │   ├── detection_service.py
│   │   └── video_service.py
│   ├── templates/
│   ├── behavior_detector.py
│   ├── models.py
│   ├── views.py
│   └── urls.py
│
├── sistema_seguridad/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── manage.py
├── db.sqlite3
└── README.md
```

---

## 🎥 Funcionamiento General del Sistema

1. 📷 Se carga un video desde una cámara o archivo
2. 🧍 YOLO detecta personas en cada frame
3. 📊 Se analiza la velocidad y cambios bruscos de movimiento
4. ⚠️ Si se detectan movimientos rápidos, forcejeos o aglomeraciones:
   - El evento se clasifica como **Sospechoso**
5. 🚨 Django registra el evento y genera una alerta
6. 📈 Los eventos pueden visualizarse en el dashboard

---

## 🚨 Criterios de Detección de Comportamiento Sospechoso

- Movimientos bruscos o acelerados
- Cambios repentinos de dirección
- Interacción violenta entre personas
- Forcejeos (robos o agresiones)
- Incremento repentino de velocidad corporal

---

## ▶️ Ejecución del Proyecto

1. Activar entorno virtual:
```bash
.venv\Scripts\activate
```

2. Ejecutar el servidor:
```bash
python manage.py runserver
```

3. Acceder desde el navegador:
```
http://127.0.0.1:8000/
```

---

## 📊 Objetivo Académico
Este proyecto fue desarrollado con fines **académicos**, demostrando el uso de **Inteligencia Artificial aplicada a la seguridad**, integrando visión por computadora y análisis de comportamiento humano.

---

## ✍️ Autora
**Scarlet Castillo**  
Proyecto académico – Sistema de Seguridad con IA

---

## ✅ Estado del Proyecto
✔ Detección de personas  
✔ Clasificación de movimiento normal / sospechoso  
✔ Detección de robos y forcejeos  
✔ Integración con Django

---

📌 *Este README puede ampliarse con instrucciones de despliegue en la nube o entrenamiento del modelo.*

