from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Phishing Detection API", description="API for detecting phishing in text, URLs, and QR codes.")

# CORS setup to allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    content: str
    type: str = "email" # or "message" or "url"

@app.get("/")
def read_root():
    return {"status": "online", "message": "Phishing Detection System Ready"}

from model import model

@app.post("/predict/text")
def predict_text(request: TextRequest):
    # Use the trained model
    is_phishing, confidence = model.predict(request.content)
    
    analysis = "Suspicious content detected." if is_phishing else "Content appears safe."
    if request.type == "url":
         analysis = "Malicious URL detected." if is_phishing else "URL appears safe."

    return {
        "is_phishing": bool(is_phishing),
        "confidence": float(confidence),
        "analysis": analysis
    }

@app.post("/decode/qr")
async def decode_qr(file: UploadFile = File(...)):
    import cv2
    import numpy as np

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
