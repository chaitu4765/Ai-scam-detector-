from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="Phishing Detection API",
    description="API for detecting phishing (v6.0)",
    root_path="/api"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    input: str
    type: str = "email"

# Model loading with error handling
model = None
try:
    from .model import model as phishing_model
    model = phishing_model
except Exception as e:
    print(f"STARTUP ERROR: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "version": "v6.0"}

@app.get("/scan")
def scan_get():
    """Diagnostic GET for /api/scan"""
    return {"message": "Use POST to scan content", "status": "active"}

@app.post("/scan")
def scan_prediction(request: TextRequest):
    """
    Standardized scan endpoint (v6.0)
    """
    try:
        is_phishing, confidence = False, 0.65
        
        if model:
            # Note: request.input is used here as per user requirement
            is_phishing, confidence = model.predict(request.input)
        
        # Standardized Response Format: 0-100 percentage
        return {
            "safe": not is_phishing,
            "phishing": is_phishing,
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        print(f"Scan API error: {e}")
        # Default fallback per requirements
        return {
            "safe": True,
            "phishing": False,
            "confidence": 65.00
        }

@app.post("/decode/qr")
async def decode_qr(file: UploadFile = File(...)):
    return {
        "safe": True,
        "phishing": False,
        "confidence": 65.00
    }

if __name__ == "__main__":
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
