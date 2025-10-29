from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import cv2
import os

# Descargar modelo si no existe
if not os.path.exists('golanalytics_vision_model.keras'):
    print("📥 Modelo no encontrado, descargando...")
    import download_model
else:
    print("✅ Modelo ya existe localmente")

app = Flask(__name__)
CORS(app)

# Cargar modelo al iniciar
print("🤖 Cargando modelo...")
model = keras.models.load_model('golanalytics_vision_model.keras')
print("✅ Modelo cargado")

# Cargar nombres de clases
with open('class_names.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)
print(f"✅ {len(class_names)} clases cargadas")

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "model": "GolAnalytics Vision",
        "classes": len(class_names),
        "message": "API funcionando correctamente"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Obtener imagen en base64
        image_base64 = data.get('image', '')
        if not image_base64:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        # Decodificar imagen
        image_data = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
        image = Image.open(BytesIO(image_data))
        
        # Convertir a formato correcto
        image = image.convert('RGB')
        image = np.array(image)
        
        # Preprocesar
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predecir
        predictions = model.predict(image, verbose=0)[0]
        
        # Obtener top 3
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "action": class_names[idx],
                "probability": float(predictions[idx])
            })
        
        return jsonify({
            "success": True,
            "predictions": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

