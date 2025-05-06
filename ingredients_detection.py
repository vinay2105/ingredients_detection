from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime as ort
from PIL import Image
import numpy as np
import io

# Initialize FastAPI app
app = FastAPI(
    title="Optimized Ingredient Detector API",
    description="Detects ingredients from a fridge image using YOLOv8 with image resizing.",
    version="1.0"
)

# Load the ONNX model
onnx_model_path = "yolov8n.onnx"  # Update with your path to the downloaded ONNX model
session = ort.InferenceSession(onnx_model_path)

# COCO class labels for YOLOv8
coco_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'none', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'none', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
    'surfboard', 'tennis racket', 'bottle', 'none', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'none', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@app.post("/detect-ingredients")
async def detect_ingredients(file: UploadFile = File(...)):
    try:
        # Read uploaded image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resize image to 640x640 for faster inference
        image = image.resize((640, 640))

        # Convert image to NumPy array for ONNX model input
        image_np = np.array(image).astype(np.float32)
        image_np = np.transpose(image_np, (2, 0, 1))  # Change from HWC to CHW format
        image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension

        # Run the ONNX model
        inputs = {session.get_inputs()[0].name: image_np}
        detections = session.run(None, inputs)[0]

        # Extract detected labels (ingredients)
        detected_ingredients = set()
        for detection in detections[0]:
            # Extract class ID from the detection (assuming class ID is in index 5)
            class_id = int(detection[5])
            label = coco_labels[class_id]  # Map class ID to label
            detected_ingredients.add(label)

        # Convert set to comma-separated string
        ingredients_string = ", ".join(detected_ingredients)

        # Return detected ingredients as JSON response
        return JSONResponse(content={"detected_ingredients": ingredients_string})

    except Exception as e:
        # Error handling
        return JSONResponse(content={"error": str(e)}, status_code=500)







