import requests
import os

print("📥 Descargando modelo desde Google Drive...")

file_id = "1Q-rJ5ZGk8DgyR5wYZDZegvfrTqMQNeou"
output = "golanalytics_vision_model.keras"

if not os.path.exists(output):
    # URL de descarga directa de Google Drive
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    
    print(f"🔗 Descargando desde: {url}")
    
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(output, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Modelo descargado correctamente")
    else:
        print(f"❌ Error al descargar: {response.status_code}")
        raise Exception(f"No se pudo descargar el modelo: {response.status_code}")
else:
    print("✅ Modelo ya existe")
