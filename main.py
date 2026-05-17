import os
import cv2
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from supabase import create_client, Client
import uvicorn

app = FastAPI(title="GolAnalytics API - YOLO + DeepSORT Tracking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase client ──────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ── YOLO model ───────────────────────────────────────────────
# YOLOv8 incluye trackers integrados: BoT-SORT y ByteTrack
# Usamos tracker.yaml personalizado para fútbol amateur
print("🤖 Cargando modelo YOLOv8n...")
model = YOLO("yolov8n.pt")
print("✅ Modelo YOLOv8n listo")


# ── Health check ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "YOLOv8n + ByteTrack",
        "service": "golanalytics-api"
    }


# ── Endpoint principal: procesar video ───────────────────────
@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    job_id: str = Form(None),
    video_id: str = Form(...),
    match_id: str = Form(...),
    team_id: str = Form(...),
):
    if not video_id or not match_id or not team_id:
        raise HTTPException(
            status_code=400,
            detail="video_id, match_id y team_id son requeridos"
        )

    # Actualizar job a 'processing'
    if job_id:
        supabase.table("tracking_jobs").update({
            "status": "processing"
        }).eq("id", job_id).execute()

    tmp_path = None
    try:
        # Guardar video en disco temporal
        suffix = os.path.splitext(file.filename)[-1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        print(f"📹 Video guardado temporalmente: {tmp_path}")

        # Abrir video con OpenCV
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="No se pudo abrir el video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Procesar a 5 frames por segundo (mejor tracking que 2fps)
        sample_interval = max(1, int(fps / 5))
        total_sampled = total_video_frames // sample_interval

        print(f"📊 FPS: {fps}, Total frames: {total_video_frames}, Muestreo cada: {sample_interval} frames")

        # Actualizar total_frames en job
        if job_id:
            supabase.table("tracking_jobs").update({
                "total_frames": total_sampled
            }).eq("id", job_id).execute()

        frame_number = 0
        processed_count = 0
        batch = []
        BATCH_SIZE = 30  # Insertar en Supabase cada 30 frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % sample_interval == 0:
                second = frame_number / fps

                # YOLO tracking con ByteTrack (tracker robusto incluido en ultralytics)
                # ByteTrack mantiene IDs más estables que el tracker básico
                # persist=True mantiene el tracker entre frames
                # tracker="bytetrack.yaml" usa configuración optimizada
                results = model.track(
                    frame,
                    classes=[0],           # solo personas
                    verbose=False,
                    persist=True,
                    tracker="bytetrack.yaml",  # Tracker robusto (mejor que botsort para fútbol)
                    conf=0.3,              # umbral de confianza
                    iou=0.5                # IoU para asociación entre frames
                )[0]

                players = []
                h, w = frame.shape[:2]

                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])

                        # track_id ahora es más estable gracias a ByteTrack
                        track_id = int(box.id[0]) if box.id is not None else -1

                        # Solo incluir detecciones con confianza >= 0.3
                        if conf >= 0.3:
                            players.append({
                                "track_id": track_id,
                                "x": round(x1 / w, 4),
                                "y": round(y1 / h, 4),
                                "width": round((x2 - x1) / w, 4),
                                "height": round((y2 - y1) / h, 4),
                                "confidence": round(conf, 3)
                            })

                batch.append({
                    "job_id": job_id,
                    "video_id": video_id,
                    "match_id": match_id,
                    "team_id": team_id,
                    "frame_number": frame_number,
                    "second_in_video": round(second, 2),
                    "players": players
                })

                processed_count += 1

                # Guardar batch en Supabase
                if len(batch) >= BATCH_SIZE:
                    supabase.table("player_tracking").insert(batch).execute()
                    batch = []
                    print(f"💾 Guardados {processed_count}/{total_sampled} frames")

                    # Actualizar progreso en job
                    if job_id:
                        supabase.table("tracking_jobs").update({
                            "processed_frames": processed_count
                        }).eq("id", job_id).execute()

            frame_number += 1

        cap.release()

        # Guardar último batch pendiente
        if batch:
            supabase.table("player_tracking").insert(batch).execute()

        # Marcar job como completado
        if job_id:
            supabase.table("tracking_jobs").update({
                "status": "completed",
                "processed_frames": processed_count,
                "completed_at": "now()"
            }).eq("id", job_id).execute()

        print(f"✅ Procesamiento completado: {processed_count} frames")

        return {
            "success": True,
            "job_id": job_id,
            "video_id": video_id,
            "total_frames_processed": processed_count,
            "fps_sampled": 5
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        # Marcar job como fallido
        if job_id:
            supabase.table("tracking_jobs").update({
                "status": "failed",
                "error_message": str(e)
            }).eq("id", job_id).execute()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Borrar video del disco siempre, sin importar si hubo error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"🗑️ Video temporal borrado: {tmp_path}")


# ── Consultar estado del job ──────────────────────────────────
@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    result = supabase.table("tracking_jobs").select("*").eq("id", job_id).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return result.data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

