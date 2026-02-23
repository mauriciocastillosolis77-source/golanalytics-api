from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import os
import uuid
import subprocess
from typing import List, Optional
from ultralytics import YOLO
import torch

# VERSION: 3.0.0 - Soccer Vision Engine (YOLOv8 + Team Color + Reception + 10 FPS)
app = FastAPI(title="GolAnalytics Vision API v3")

# Modelo YOLOv8m (medium) - Optimizado para detección de objetos pequeños (balón)
model = YOLO("yolov8m.pt")

class AnalysisRequest(BaseModel):
    video_url: str
    start_time: str  # MM:SS
    end_time: str    # MM:SS

class VisionEvent(BaseModel):
    timestamp: str
    from_player: Optional[int] = None
    to_player: Optional[int] = None
    team: str = "unknown"
    event: str
    confidence: float

def time_to_seconds(t_str):
    parts = t_str.split(':')
    return int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else 0

def seconds_to_timestamp(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 100)
    return f"{m:02d}:{s:02d}.{ms:02d}"

def get_team_color(frame, box):
    """Clasificación por color de uniforme usando análisis HSV en el torso"""
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y1+int((y2-y1)*0.4), x1:x2]
    if roi.size == 0: return "unknown"
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_roi, lower_green, upper_green)
    non_green_roi = cv2.bitwise_and(hsv_roi, hsv_roi, mask=cv2.bitwise_not(mask))
    
    pixels = non_green_roi.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
    
    if len(pixels) == 0: return "unknown"
    
    avg_h = np.mean(pixels[:, 0])
    avg_s = np.mean(pixels[:, 1])
    
    if avg_s < 40: return "team_white"
    if avg_h < 15 or avg_h > 165: return "team_red"
    if 95 < avg_h < 135: return "team_blue"
    if 20 < avg_h < 38: return "team_yellow"
    return "team_other"

@app.post("/vision/analyze", response_model=List[VisionEvent])
async def analyze_video(request: AnalysisRequest):
    job_id = str(uuid.uuid4())
    temp_video = f"temp_{job_id}.mp4"
    start_s = time_to_seconds(request.start_time)
    end_s = time_to_seconds(request.end_time)
    duration = max(0, end_s - start_s)
    
    try:
        subprocess.run(['ffmpeg', '-ss', str(start_s), '-t', str(duration), '-i', request.video_url, '-c', 'copy', temp_video, '-y'], check=True, capture_output=True)
        
        events = []
        cap = cv2.VideoCapture(temp_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        last_possession_id = None
        last_team = "unknown"
        
        process_every = max(1, int(fps / 10))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % process_every == 0:
                current_time_rel = start_s + (frame_count / fps)
                results = model.track(frame, persist=True, classes=[0, 32], verbose=False, conf=0.25)
                
                boxes = results[0].boxes
                ball_box = None
                players = []
                
                for box in boxes:
                    cls, conf = int(box.cls[0]), float(box.conf[0])
                    b = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id[0]) if box.id is not None else -1
                    if cls == 32: ball_box = b
                    elif cls == 0: players.append({"id": track_id, "box": b, "conf": conf})
                
                if ball_box is not None:
                    ball_center = np.array([(ball_box[0]+ball_box[2])/2, (ball_box[1]+ball_box[3])/2])
                    closest_player, min_dist = None, float('inf')
                    
                    for p in players:
                        p_center = np.array([(p["box"][0]+p["box"][2])/2, (p["box"][1]+p["box"][3])/2])
                        dist = np.linalg.norm(ball_center - p_center)
                        if dist < 55 and dist < min_dist:
                            min_dist, closest_player = dist, p
                    
                    if closest_player:
                        p_id = closest_player["id"]
                        current_team = get_team_color(frame, closest_player["box"])
                        
                        if last_possession_id is not None and last_possession_id != p_id:
                            events.append(VisionEvent(
                                timestamp=seconds_to_timestamp(current_time_rel),
                                from_player=last_possession_id,
                                to_player=p_id,
                                team=current_team,
                                event="reception",
                                confidence=closest_player["conf"]
                            ))
                            events.append(VisionEvent(
                                timestamp=seconds_to_timestamp(max(start_s, current_time_rel - 0.2)),
                                from_player=last_possession_id,
                                to_player=p_id,
                                team=last_team,
                                event="pass",
                                confidence=0.85
                            ))
                        elif last_possession_id is None:
                            events.append(VisionEvent(
                                timestamp=seconds_to_timestamp(current_time_rel),
                                from_player=p_id,
                                team=current_team,
                                event="possession",
                                confidence=closest_player["conf"]
                            ))
                        last_possession_id, last_team = p_id, current_team
            frame_count += 1
            
        cap.release()
        if os.path.exists(temp_video): os.remove(temp_video)
        return events
    except Exception as e:
        if os.path.exists(temp_video): os.remove(temp_video)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
