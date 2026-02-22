FROM python:3.10-slim

# Dependencias para OpenCV y procesamiento de video
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_v2_0_0.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Descarga el modelo durante la construcción de la imagen para despliegues rápidos
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

COPY app_v2_0_0.py ./app.py

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
