from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.model import load_model, inference
import base64
import os

app = FastAPI(title="Vehicle Attribute Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-project.web.app", "http://localhost:3000"],  # Replace with your Firebase URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model once at startup
model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs inference, 
    and returns both the detection summary and the annotated image (base64 encoded).
    """
    try:
        image_bytes = await file.read()
        results = inference(model, image_bytes)
        annotated_path = results.get("annotated_image_path")
        annotated_b64 = None
        if annotated_path and os.path.exists(annotated_path):
            with open(annotated_path, "rb") as f:
                annotated_b64 = base64.b64encode(f.read()).decode("utf-8")

        response = {
            "num_detections": results["summary"].get("num_detections", 0),
            "detections": results["summary"].get("detections", []),
            "annotated_image_base64": annotated_b64
        }
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)