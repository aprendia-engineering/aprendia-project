# Sign Language Recognition

Sistema de reconocimiento de lenguaje de signos usando inteligencia artificial.

## Estructura del Proyecto

```
sign-language-recognition/
├── data/
│   ├── raw/              # Videos originales grabados
│   ├── processed/        # landmarks.csv + .npy
│   └── labels.json       # Mapeo de señas → clase
├── src/
│   ├── tracking.py       # MediaPipe + OpenCV
│   ├── landmarks.py      # Extracción y normalización
│   ├── classifier.py     # Carga y predicción del modelo
│   ├── collector.py      # Recolección de dataset
│   ├── visualizer.py     # Overlay de landmarks en frame
│   └── __init__.py
├── models/
│   └── sign_model.h5     # Modelo Keras entrenado
├── tests/
│   ├── test_tracking.py
│   └── test_landmarks.py
├── main.py               # Punto de entrada principal
└── requirements.txt      # Dependencias del proyecto
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```
