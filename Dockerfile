FROM python:3.10-slim

# Dependencias para OpenCV y procesamiento de video
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descarga el modelo durante la construcción de la imagen
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Instalamos uvicorn explícitamente
RUN pip install --no-cache-dir uvicorn gunicorn

COPY app.py .

# Variable de entorno PORT manejada por Railway
ENV PORT=8080

# Forzamos el uso de python -m uvicorn y deshabilitamos cualquier autodetección de Railway
# Al usar una cadena simple en CMD sin corchetes, se ejecuta en un shell que procesa $PORT
CMD python -m uvicorn app:app --host 0.0.0.0 --port $PORT

