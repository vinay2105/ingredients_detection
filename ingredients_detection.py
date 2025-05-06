from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Initialize FastAPI app
app = FastAPI(
    title="Ingredient Detector API",
    description="Detects ingredients in a fridge image using YOLOv8.",
    version="1.0"
)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

@app.post("/detect-ingredients")
async def detect_ingredients(file: UploadFile = File(...)):
    try:
        # Save uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image = Image.open(file.file)
            image.save(temp_file.name)
            temp_image_path = temp_file.name

        # Run YOLOv8 detection
        results = model.predict(temp_image_path, save=False)

        # Extract detected ingredients
        detected_ingredients = set()
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_ingredients.add(label)

        # Clean up temp image
        os.remove(temp_image_path)

        # Convert set to comma-separated string
        ingredients_string = ", ".join(detected_ingredients)

        # Return detected ingredients string in JSON
        return JSONResponse(content={
            "detected_ingredients": ingredients_string
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
