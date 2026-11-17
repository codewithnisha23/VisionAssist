# VisionAssist — AI-Powered Visual Guide

A React + FastAPI academic web application based on the supplied VisionAssist synopsis and design references.

## Features

- Responsive dark navy dashboard with blue, cyan, purple and teal feature cards.
- Browser camera capture and image upload.
- Object detection using YOLOv8 (`yolov8n.pt` by default).
- Text extraction using EasyOCR.
- Text-to-speech feedback using the browser Speech Synthesis API (English and Hindi voice selection; voice availability depends on the device).
- Currency recognition API integration for a separately trained Indian banknote YOLO model.
- Accessibility controls for audio feedback and larger text.

## Project structure

```text
visionassist-app/
├── src/                    # React frontend
│   ├── main.jsx
│   └── style.css
├── backend/
│   ├── app/main.py          # FastAPI endpoints
│   └── requirements.txt
├── index.html
├── package.json
└── vite.config.js
```

## Requirements

- Node.js 18 or newer and npm
- Python 3.10–3.12 recommended
- A webcam for live camera input (or upload images)

## Run the frontend

Open a terminal in the project root:

```bash
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). Camera access requires localhost or HTTPS and browser permission.

## Run the backend

Open a second terminal:

```bash
cd backend
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows CMD**
```bat
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and start the API:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation is available at `http://localhost:8000/docs`.

## AI model notes — important

### Object detection

The backend uses Ultralytics YOLO. On the first object-detection request, `yolov8n.pt` may be downloaded automatically. An internet connection may be needed for the first run. You can point `YOLO_MODEL` to another compatible object detection model.

### OCR

EasyOCR loads its English model on the first request and may download model weights. Clear, well-lit images produce better results. This starter configuration uses English OCR.

### Indian currency recognition

A reliable currency-denomination detector requires a model trained on Indian banknote images. The supplied synopsis does not include trained weights or a labelled dataset. Therefore the endpoint is implemented and will return a clear “model not configured” message until you provide a compatible trained YOLO model. Set the `CURRENCY_MODEL` environment variable to its `.pt` path before starting FastAPI. Do not use prototype predictions to make financial decisions.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | API and model status |
| POST | `/api/detect` | Detect objects in an uploaded image |
| POST | `/api/read-text` | Extract text using EasyOCR |
| POST | `/api/recognize-currency` | Identify a note using your trained currency model |

All POST endpoints accept multipart form data with an image field named `file`.

## Frontend API URL

By default the frontend calls `http://localhost:8000`. If your backend runs elsewhere, create a `.env` file in the project root:

```env
VITE_API_URL=http://localhost:8000
```

Restart the Vite server after changing the environment file.

## Safety and academic prototype notice

This is a student project prototype, not a certified mobility or navigation aid. It does not provide dependable obstacle avoidance, distance measurement, route guidance, or guaranteed currency authentication. Users should continue using a white cane, guide dog, trusted mobility tools, and/or a human assistant where appropriate. Never rely on the app as the sole source of safety information.

## Based on the supplied synopsis

The project implements the synopsis modules: User Interface, Camera Input, Object Detection, OCR, Currency Recognition, and Audio Guidance. The frontend visual style follows the provided dashboard reference and related screen examples.

## Real-time Indian currency scanning

The Currency Recognition screen now supports continuous webcam scanning: start the camera, then select **Start Real-time Recognition**. The browser sends a frame about every 1.4 seconds to `POST /api/recognize-currency`; the backend keeps the YOLO model loaded between frames and speaks a newly detected denomination once.

**Important:** continuous scanning is implemented, but reliable denomination detection requires trained Indian-banknote YOLO weights. This repository does not include those weights or a labelled training dataset, so the API intentionally reports `Not identified` until you provide a model. Do not treat predictions as proof of denomination or use this prototype as the sole aid for financial transactions.

To configure trained weights:
1. Place your trained YOLO weights file (for example `best.pt`) in `backend/models/`.
2. Copy `.env.example` to `.env` and set `CURRENCY_MODEL=models/best.pt` (or set the environment variable to the absolute path).
3. Restart FastAPI. Your model's class names should clearly map to denominations such as `10_rupees`, `20_rupees`, `50_rupees`, `100_rupees`, `200_rupees`, and `500_rupees`.
4. Train and validate on varied front/back note images, angles, distances, backgrounds and lighting before any user testing. Include a held-out test set and report per-denomination precision/recall.

Camera access requires localhost or HTTPS and browser permission. The live scan is a frame-by-frame prototype, not a guaranteed safety-critical real-time system.
