from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI(
    title="Ingredient Detector API",
    description="Detects ingredients from a fridge image using YOLOv8 with ONNX.",
    version="1.0"
)

# Load the ONNX model (YOLOv8) into ONNX Runtime
onnx_model_path = "yolov8n.onnx"  # Path to the locally downloaded ONNX model
session = ort.InferenceSession(onnx_model_path)

# Extract the input name from the model
input_name = session.get_inputs()[0].name

@app.post("/detect-ingredients")
async def detect_ingredients(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize the image to 640x640 for YOLOv8 (standard input size)
        image = image.resize((640, 640))
        img_np = np.array(image).astype(np.float32)

        # Normalize the image to [0, 1] range and convert from HWC to CHW format
        img_np = img_np / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
        img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension

        # Run inference on the image
        outputs = session.run(None, {input_name: img_np})

        # Parse the output (this part depends on the YOLOv8 ONNX output format)
        detections = outputs[0]  # Detection outputs (could be in format [batch, detections, 6])
        detected_labels = []

        # Iterate through detections and extract detected labels
        for i in range(detections.shape[1]):
            # Extract the class ID of the detection (assuming class ID is in index 5)
            class_id = int(detections[0, i, 5])
            label = session.get_meta().get('labels')[class_id]  # Assuming labels are in the model metadata
            detected_labels.append(label)

        # Return the detected labels as a response
        return JSONResponse(content={
            "message": f"Model ran successfully. {len(detected_labels)} detections found.",
            "detected_labels": detected_labels
        })

    except Exception as e:
        # Handle errors and exceptions
        return JSONResponse(content={"error": str(e)}, status_code=500)






