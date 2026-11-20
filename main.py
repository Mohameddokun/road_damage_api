import base64
import os
import time
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from tracker import DamageTracker, DAMAGE_CLASSES
from utils import assess_road_condition, draw_annotations
import threading
from detector import RoadDamageDetector

CLASS_ID_TO_CODE = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
}

detector = RoadDamageDetector(
    model_path="weights/best.pt",
    conf_threshold=0.33,
    imgsz=800,
)
inference_lock = threading.Lock()


def run_inference(frame: np.ndarray) -> list[dict]:
    """
    Calls RoadDamageDetector.predict_frame() (CLAHE + conf=0.33 + imgsz=800)
    and normalizes its output for DamageTracker and draw_annotations().
    """
    with inference_lock:
        _, raw_detections = detector.predict_frame(frame)

    return [
        {
            "bbox": det["bbox"],
            "class_name": CLASS_ID_TO_CODE.get(det["class_id"], det["label"].split(" ")[0]),
            "label": det["label"],
            "conf": det["confidence"],
        }
        for det in raw_detections
    ]
# --------------------------------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Road Damage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.post("/predict/image")
def predict_image(file: UploadFile = File(...)):
    raw = file.file.read()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    t0 = time.perf_counter()
    detections = run_inference(frame)
    elapsed = max(time.perf_counter() - t0, 1e-6)
    fps = 1.0 / elapsed

    counts = {cls: 0 for cls in DAMAGE_CLASSES}
    for det in detections:
        if det["class_name"] in counts:
            counts[det["class_name"]] += 1

    condition = assess_road_condition(counts)
    annotated = draw_annotations(frame, detections, fps, condition)

    filename = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), annotated)

    return {
        "counts": counts,
        "fps": round(fps, 2),
        "road_condition": condition,
        "output_url": f"/outputs/{filename}",
    }


@app.post("/predict/video")
def predict_video(file: UploadFile = File(...)):
    temp_in = os.path.join(OUTPUT_DIR, f"in_{uuid.uuid4().hex}.mp4")
    out_filename = f"out_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_filename)

    with open(temp_in, "wb") as f:
        f.write(file.file.read())

    cap = cv2.VideoCapture(temp_in)
    if not cap.isOpened():
        os.remove(temp_in)
        raise HTTPException(status_code=400, detail="Could not open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Use 'mp4v' for broad OpenCV compatibility (or 'avc1' if H.264 is installed for browser playback)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, src_fps, (width, height))

    tracker = DamageTracker(iou_threshold=0.3, max_lost=12, min_hits=2)
    frame_count = 0
    t_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_frame = time.perf_counter()
            raw_dets = run_inference(frame)
            tracked_dets = tracker.update(raw_dets)
            inst_fps = 1.0 / max(time.perf_counter() - t_frame, 1e-6)

            counts = tracker.get_counts()
            condition = assess_road_condition(counts)
            annotated = draw_annotations(frame, tracked_dets, inst_fps, condition)

            writer.write(annotated)
            frame_count += 1
    finally:
        cap.release()
        writer.release()
        if os.path.exists(temp_in):
            os.remove(temp_in)

    total_time = max(time.perf_counter() - t_start, 1e-6)
    avg_fps = frame_count / total_time
    final_counts = tracker.get_counts()

    return {
        "counts": final_counts,
        "fps": round(avg_fps, 2),
        "road_condition": assess_road_condition(final_counts),
        "frames_processed": frame_count,
        "video_url": f"/outputs/{out_filename}",
    }


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Receives raw JPEG bytes from a browser/mobile camera over WebSocket,
    returns JSON with base64-encoded annotated frame + live stats.
    """
    await websocket.accept()
    tracker = DamageTracker(iou_threshold=0.3, max_lost=10, min_hits=2)

    try:
        while True:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.perf_counter()
            tracked_dets = tracker.update(run_inference(frame))
            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)

            counts = tracker.get_counts()
            condition = assess_road_condition(counts)
            annotated = draw_annotations(frame, tracked_dets, fps, condition)

            _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            await websocket.send_json({
                "counts": counts,
                "fps": round(fps, 2),
                "road_condition": condition,
                "frame": base64.b64encode(buf).decode("utf-8"),
            })
    except WebSocketDisconnect:
        pass


# Optional: Direct local webcam stream for quick demos (open http://127.0.0.1:8000/live/mjpeg)
live_stats = {"counts": {c: 0 for c in DAMAGE_CLASSES}, "fps": 0.0, "road_condition": "Good"}


def generate_webcam_frames(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    tracker = DamageTracker(iou_threshold=0.3, max_lost=10, min_hits=2)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            tracked_dets = tracker.update(run_inference(frame))
            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)

            counts = tracker.get_counts()
            condition = assess_road_condition(counts)
            live_stats.update({"counts": counts, "fps": round(fps, 2), "road_condition": condition})

            annotated = draw_annotations(frame, tracked_dets, fps, condition)
            _, jpg = cv2.imencode(".jpg", annotated)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.get("/live/mjpeg")
def live_mjpeg(camera_index: int = 0):
    return StreamingResponse(
        generate_webcam_frames(camera_index),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/live/stats")
def get_live_stats():
    return live_stats