from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pathlib import Path
import os
import numpy as np
from functools import lru_cache

app = FastAPI(title="VisionAssist API", version="1.0.0", description="AI assistance API for VisionAssist")
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

frontend_origins = os.getenv("FRONTEND_ORIGINS", "")

allowed_origins += [
    origin.strip()
    for origin in frontend_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MAX_BYTES = 12 * 1024 * 1024

@lru_cache(maxsize=2)
def get_yolo_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)

def load_image(raw: bytes):
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum upload size is 12 MB.")
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
        if image.width < 40 or image.height < 40:
            raise HTTPException(status_code=400, detail="Image is too small. Please upload a clearer image.")
        return image
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a supported image.")

@app.get("/")
def root():
    return {"name": "VisionAssist API", "status": "running", "docs": "/docs"}

@app.get("/api/health")
def health():
    return {"status": "ok", "object_model": "configured" if Path(os.getenv("YOLO_MODEL", "yolov8n.pt")).exists() else "download on first use", "currency_model": "configured" if os.getenv("CURRENCY_MODEL") and Path(os.getenv("CURRENCY_MODEL")).exists() else "not configured"}

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    image = load_image(await file.read())
    try:
        from ultralytics import YOLO
        model_path = os.getenv("YOLO_MODEL", "yolov8n.pt")
        model = get_yolo_model(model_path)
        results = model.predict(np.array(image), conf=0.35, verbose=False)
        names = results[0].names
        objects = []
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            objects.append({"name": str(names[cls]), "confidence": round(float(box.conf[0].item()), 3)})
        # Count repeated labels to make the spoken summary natural.
        counts = {}
        for obj in objects: counts[obj["name"]] = counts.get(obj["name"], 0) + 1
        summary = "Detected " + ", ".join(f"{count} {name}{'s' if count > 1 else ''}" for name, count in counts.items()) if counts else "No familiar objects were detected."
        return {"objects": objects, "summary": summary}
    except ImportError:
        raise HTTPException(status_code=503, detail="Object detection dependencies are not installed. Run pip install -r requirements.txt.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(exc)}")

@app.post("/api/read-text")
async def read_text(file: UploadFile = File(...)):
    image = load_image(await file.read())
    try:
        import easyocr
        # Lazy load: first request downloads OCR model weights if they are not cached.
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        lines = reader.readtext(np.array(image), detail=0, paragraph=True)
        text = "\n".join(str(line) for line in lines).strip()
        return {"text": text, "found": bool(text)}
    except ImportError:
        raise HTTPException(status_code=503, detail="OCR dependencies are not installed. Run pip install -r requirements.txt.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text reading failed: {str(exc)}")

@app.post("/api/recognize-currency")
async def recognize_currency(file: UploadFile = File(...)):
    image = load_image(await file.read())
    model_path = os.getenv("CURRENCY_MODEL", "")
    if not model_path or not Path(model_path).exists():
        return {"denomination": "Not identified", "message": "The currency recognition module is ready, but a trained Indian banknote detection model is not configured yet.", "note": "For real denomination predictions, add your trained YOLO currency weights and set CURRENCY_MODEL to its path. Do not rely on this prototype for financial transactions."}
    try:
        from ultralytics import YOLO
        model = get_yolo_model(model_path)
        results = model.predict(np.array(image), conf=0.4, verbose=False)
        names = results[0].names
        detections = [{"name": str(names[int(b.cls[0].item())]), "confidence": float(b.conf[0].item())} for b in results[0].boxes]
        if not detections:
            return {"denomination": "Not identified", "message": "No currency note was confidently detected. Try a clear, well-lit image with the full note visible."}
        best = max(detections, key=lambda x: x["confidence"])
        return {"denomination": best["name"], "message": f"Possible currency note: {best['name']} ({best['confidence']:.0%} confidence). Please verify manually.", "confidence": round(best["confidence"], 3)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Currency recognition failed: {str(exc)}")
