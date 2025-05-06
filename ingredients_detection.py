from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

# Initialize FastAPI app
app = FastAPI(
    title="Ingredient Detector API",
    description="Detects ingredients from an image using a local YOLOv8 ONNX model.",
    version="1.0"
)

# Load ONNX model once at startup
model_path = "yolov8n.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Model input details
input_name = session.get_inputs()[0].name

@app.post("/detect-ingredients")
async def detect_ingredients(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((640, 640))  # Resize to model's expected input size
        img_np = np.array(image).astype(np.float32)

        # Preprocess: normalize to 0-1 and shape to (1, 3, 640, 640)
        img_np = img_np / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
        img_np = np.expand_dims(img_np, axis=0)

        # Run inference
        outputs = session.run(None, {input_name: img_np})

        # Parse output (this depends on your ONNX model structure)
        # Here, we'll just show shape or sample outputs for demo
        detections = outputs[0]
        detection_count = detections.shape[1]

        return JSONResponse(content={
            "message": f"Model ran successfully. {detection_count} detections found."
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)





