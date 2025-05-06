from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import onnxruntime
import io

# Initialize FastAPI app
app = FastAPI(
    title="Optimized Ingredient Detector API",
    description="Detects ingredients from a fridge image using YOLOv8 with image resizing.",
    version="1.0"
)

# Load ONNX model once at startup
session = onnxruntime.InferenceSession('yolov8n.onnx')

# Define the image preprocessing function
def preprocess_image(image: Image):
    # Resize to 640x640 (input size for YOLO)
    image = image.resize((640, 640))
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
    return img_array

@app.post("/detect-ingredients")
async def detect_ingredients(file: UploadFile = File(...)):
    try:
        # Read uploaded image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess image
        img_array = preprocess_image(image)

        # Run inference using onnxruntime
        inputs = {session.get_inputs()[0].name: img_array}
        outputs = session.run(None, inputs)

        # Process output (simplified for this example, depends on your model's output format)
        detected_classes = outputs[0]  # This will contain the predictions

        # Example to map outputs to labels (this depends on your model)
        # We assume the output is a list of predicted class indices
        # You may need to adjust this according to the actual model output format
        detected_ingredients = [f"Class {cls}" for cls in detected_classes]

        return {"detected_ingredients": detected_ingredients}

    except Exception as e:
        # Error handling
        return JSONResponse(content={"error": str(e)}, status_code=500)



