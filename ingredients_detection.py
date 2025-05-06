from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# Initialize FastAPI app
app = FastAPI(
    title="Optimized Ingredient Detector API",
    description="Detects ingredients from a fridge image using YOLOv8 with image resizing.",
    version="1.0"
)

# Load YOLOv8 model once at startup (using yolov8n.pt for lightweight inference)
model = YOLO('yolov8n.pt')

@app.post("/detect-ingredients")
async def detect_ingredients(file: UploadFile = File(...)):
    try:
        # Read uploaded image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resize image to 640x640 for faster inference
        image = image.resize((640, 640))

        # Convert image to NumPy array for YOLOv8
        image_np = np.array(image)

        # Run YOLOv8 detection directly on NumPy array, disable save/verbose
        results = model.predict(source=image_np, save=False, verbose=False)

        # Extract detected labels (ingredients)
        detected_ingredients = set()
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_ingredients.add(label)

        # Convert set to comma-separated string
        ingredients_string = ", ".join(detected_ingredients)

        # Return detected ingredients as JSON response
        return JSONResponse(content={"detected_ingredients": ingredients_string})

    except Exception as e:
        # Error handling
        return JSONResponse(content={"error": str(e)}, status_code=500)

