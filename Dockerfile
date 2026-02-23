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

# Descarga el modelo durante la construcción de la imagen para despliegues rápidos
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Aseguramos que uvicorn esté instalado
RUN pip install --no-cache-dir uvicorn

COPY app.py .

# Railway asigna un puerto dinámico a la variable $PORT
ENV PORT=8080
EXPOSE 8080

# Usamos sh -c para que la variable $PORT sea interpretada correctamente por el shell
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
