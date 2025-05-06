from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import onnxruntime
import io
import os
import urllib.request

# Model file config
model_url = "https://your-hosted-link.com/yolov8n.onnx"  # 🔄 Replace with your real model URL
model_path = "yolov8n.onnx"

# Download model if not present
if not os.path.exists(model_path):
    print("Downloading YOLOv8 ONNX model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model download complete.")

# Initialize ONNX model session
session = onnxruntime.InferenceSession(model_path)

# Initialize FastAPI app
app = FastAPI(
    title="Optimized Ingredient Detector API",
    description="Detects ingredients from a fridge image using YOLOv8 ONNX with image resizing.",
    version="1.0"
)

# Image preprocessing
def preprocess_image(image: Image):
    image = image.resize((640, 640))
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
    return img_array

# Prediction endpoint
@app.post("/detect-ingredients")
async def detect_ingredients(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess
        img_array = preprocess_image(image)

        # Run inference
        inputs = {session.get_inputs()[0].name: img_array}
        outputs = session.run(None, inputs)

        # Process outputs — for this example, we’ll just return raw output
        detected_classes = outputs[0].tolist()

        return JSONResponse(content={"detected_classes_raw": detected_classes})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)




