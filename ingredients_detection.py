from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI(title="Fast Ingredient Detector API")

# Enable CORS for all origins (to connect frontend if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8n model once at startup
model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # Read uploaded image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize image for faster inference
        image = image.resize((640, 640))

        # Run YOLO detection
        results = model.predict(source=image, save=False, verbose=False)

        # Extract detected labels
        labels = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            labels.append(model.names[cls_id])

        return JSONResponse(content={"detected_ingredients": list(set(labels))})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)








