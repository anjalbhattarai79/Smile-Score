import cv2
from deepface import DeepFace
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anjalbhattarai79.github.io/Smile-Score/"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#Root endpoint
@app.get("/")
async def read_root():
    return {"message": "api loaded successfully"}

#Receive image to predict emotion
@app.post("/predict_emotion/")
async def predict_emotion_from_image(image: UploadFile = File(...)):
    try:
        # Read uploaded file bytes
        image_bytes = await image.read()

        # Convert bytes → numpy array → OpenCV image (BGR)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
        # Analyze the image using DeepFace
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    
        # Extract happiness score
        emotion_scores = analysis[0]['emotion']
        happiness_score = emotion_scores['happy']

        return JSONResponse(status_code=201, content={"happiness_score": f"Happiness Score: {happiness_score:.2f}"} )
    
    except Exception as e:
        return str(e)


