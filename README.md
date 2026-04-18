# AprendIA1

**Reconocimiento y reconstrucción de lenguaje de signos mediante inteligencia artificial**

AprendIA1 es un proyecto académico que utiliza visión por computadora e inteligencia artificial para capturar, analizar y reconstruir movimientos de manos en lengua de signos. El sistema extrae puntos clave del cuerpo usando MediaPipe, procesa estos datos con modelos de deep learning y los presenta en visualizaciones interactivas para mejorar la accesibilidad.

## 🎯 Objetivos

- Capturar y digitalizar movimientos de lengua de signos desde video
- Extraer y normalizar características de manos y postura usando MediaPipe
- Entrenar y evaluar modelos de clasificación de señas
- Visualizar landmarks en tiempo real
- Facilitar la reconstrucción de gestos en avatares digitales

## 📁 Estructura del Proyecto

```
AprendIA1/
├── data/
│   ├── raw/              # Videos originales sin procesar
│   ├── processed/        # Datos extraídos (landmarks.csv + .npy)
│   └── labels.json       # Mapeo de señas → clases
├── src/
│   ├── tracking.py       # Detección de poses (MediaPipe + OpenCV)
│   ├── landmarks.py      # Extracción y normalización de features
│   ├── classifier.py     # Modelo de clasificación de señas
│   ├── collector.py      # Herramientas de recolección de dataset
│   ├── visualizer.py     # Visualización de landmarks en video
│   └── __init__.py
├── models/
│   └── sign_model.h5     # Modelo Keras entrenado
├── tests/
│   ├── test_tracking.py
│   └── test_landmarks.py
├── main.py               # Script principal
└── requirements.txt      # Dependencias
```

## ⚙️ Requisitos

- **Python 3.11.9** (o superior)
- pip (gestor de paquetes)
- OpenCV
- MediaPipe
- Keras/TensorFlow

## 🚀 Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd AprendIA1
```

2. Crear un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 💻 Uso

### Ejecutar el sistema completo:
```bash
python main.py
```

### Módulos disponibles:

- **Tracking**: Detección de poses en video
- **Landmarks**: Extracción de puntos clave
- **Classifier**: Predicción de señas
- **Collector**: Captura de nuevas muestras de entrenamiento
- **Visualizer**: Visualización interactiva

## 🧪 Tests

Ejecutar pruebas unitarias:
```bash
pytest tests/
```

## 📝 Licencia

Ver [LICENSE](LICENSE) para más información.


