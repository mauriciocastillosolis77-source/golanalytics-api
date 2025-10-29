import gdown
import os

print("📥 Descargando modelo desde Google Drive...")

# ID del archivo (extraído del link)
file_id = "1Q-rJ5ZGk8DgyR5wYZDZegvfrTqMQNeou"

# URL de descarga directa
url = f"https://drive.google.com/uc?id={file_id}"

# Descargar
output = "golanalytics_vision_model.keras"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
    print("✅ Modelo descargado correctamente")
else:
    print("✅ Modelo ya existe")
