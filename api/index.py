from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="Phishing Detection API",
    description="API for detecting phishing in text, URLs, and QR codes (Standardized v1.6)",
    root_path="/api"
)

# CORS setup to allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import sys

# Ensure the current directory is in sys.path for relative imports to work as absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

startup_error = None
model = None

try:
    from model import model as phishing_model
    model = phishing_model
except Exception as e:
    startup_error = str(e)
    print(f"STARTUP ERROR: {e}")

@app.get("/")
def read_root():
    return {
        "status": "online" if not startup_error else "error",
        "version": "v2.3",
        "startup_error": startup_error,
        "model_errors": model.errors if model else ["Model object not created"],
        "current_dir": current_dir,
        "sys_path": sys.path[:5], # Show first few entries for debugging
        "files_in_dir": os.listdir(current_dir)
    }

@app.post("/predict/text")
def predict_text(request: TextRequest):
    # Use the trained model
    is_phishing, confidence = False, 0.0
    errors = []
    
    if model:
        is_phishing, confidence = model.predict(request.content)
        errors = model.errors
    else:
        errors = ["Model not initialized"]

    analysis = "Suspicious content detected." if is_phishing else "Content appears safe."
    if request.type == "url":
         analysis = "Malicious URL detected." if is_phishing else "URL appears safe."
    
    # If confidence is 0, append a warning about model loading
    if confidence == 0:
        analysis += f" (Warning: AI model might not be loaded. Errors: {', '.join(errors)})"

    return {
        "is_phishing": bool(is_phishing),
        "confidence": float(confidence),
        "analysis": analysis,
        "debug_errors": errors
    }

@app.post("/decode/qr")
async def decode_qr(file: UploadFile = File(...)):
    try:
        import cv2
        import numpy as np
    except ImportError:
         return {
             "is_phishing": False,
             "confidence": 0.0,
             "analysis": "QR Code feature disabled on this deployment due to size limits."
         }

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect QR code
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)

    if data:
        # If QR code found, check the data (URL)
        return predict_text(TextRequest(content=data, type="url"))
    else:
        return {
             "is_phishing": False,
             "confidence": 0.0,
             "analysis": "No QR code detected or could not decode."
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
