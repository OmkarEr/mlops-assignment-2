import logging
import time
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# M5: Basic Monitoring & Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global metrics counters
REQUEST_COUNT = 0

try:
    model = tf.keras.models.load_model("model.h5")
except:
    model = None

@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    start_time = time.time()
    
    logger.info(f"Prediction requested for file: {file.filename} (Total Requests: {REQUEST_COUNT})")
    
    image = Image.open(io.BytesIO(await file.read())).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if model:
        pred = model.predict(img_array)
        class_label = "Dog" if pred[0][0] > 0.5 else "Cat"
        
        latency = time.time() - start_time
        logger.info(f"Prediction complete in {latency:.4f}s. Result: {class_label}")
        
        return {
            "label": class_label, 
            "probability": float(pred[0][0]),
            "latency": latency,
            "total_requests": REQUEST_COUNT
        }
        
    return {"error": "Model not found. Run train.py first."}