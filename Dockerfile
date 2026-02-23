FROM python:3.10-slim

# Dependencias para OpenCV y procesamiento de video
# Se cambió libgl1-mesa-glx por libgl1 para compatibilidad con debian trixie/slim (v4_0_0)
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

COPY app.py .

EXPOSE 8080

# Usamos python -m uvicorn para asegurar que encuentre el ejecutable en el PATH
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
