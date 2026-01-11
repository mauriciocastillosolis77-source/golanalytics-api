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
model_path = 'golanalytics_vision_model.keras'
# Descargar modelo si no existe
if not os.path.exists(model_path):
    print("📥 Modelo no encontrado, descargando...")
    import download_model
else:
    print("✅ Modelo ya existe localmente")
app = Flask(__name__)
CORS(app)
# Cargar modelo al iniciar
print("🤖 Cargando modelo...")
model = keras.models.load_model(model_path)
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
def preprocess_image(image_base64):
    """Helper function to preprocess a single image"""
    # Decodificar imagen
    image_data = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
    image = Image.open(BytesIO(image_data))
    
    # Convertir a formato correcto
    image = image.convert('RGB')
    image = np.array(image)
    
    # Preprocesar
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    
    return image
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Obtener imagen en base64
        image_base64 = data.get('image', '')
        if not image_base64:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        # Preprocesar imagen
        image = preprocess_image(image_base64)
        image = np.expand_dims(image, axis=0)
        
        # Predecir
        predictions = model.predict(image, verbose=0)[0]
        
        # Obtener top 3
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "action": class_names[str(idx)],
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
@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    try:
        data = request.get_json()
        
        # Obtener array de frames
        frames = data.get('frames', [])
        if not frames or not isinstance(frames, list):
            return jsonify({"success": False, "error": "No frames provided or invalid format"}), 400
        
        if len(frames) > 50:
            return jsonify({"success": False, "error": "Maximum 50 frames per batch"}), 400
        
        # Preprocesar todas las imágenes
        processed_images = []
        valid_indices = []
        
        for idx, frame_data in enumerate(frames):
            try:
                image_base64 = frame_data.get('image', '')
                if image_base64:
                    image = preprocess_image(image_base64)
                    processed_images.append(image)
                    valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing frame {idx}: {str(e)}")
                continue
        
        if not processed_images:
            return jsonify({"success": False, "error": "No valid frames to process"}), 400
        
        # Convertir a batch
        batch = np.array(processed_images)
        
        # Predecir en batch (más eficiente)
        predictions_batch = model.predict(batch, verbose=0)
        
        # Procesar resultados
        results = []
        for idx, predictions in zip(valid_indices, predictions_batch):
            # Obtener top 3
            top_indices = np.argsort(predictions)[-3:][::-1]
            
            frame_results = []
            for pred_idx in top_indices:
                frame_results.append({
                    "action": class_names[str(pred_idx)],
                    "probability": float(predictions[pred_idx])
                })
            
            results.append({
                "frame_index": idx,
                "timestamp": frames[idx].get('timestamp', 0),
                "predictions": frame_results
            })
        
        return jsonify({
            "success": True,
            "total_frames": len(frames),
            "processed_frames": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
