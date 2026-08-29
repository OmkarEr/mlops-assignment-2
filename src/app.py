import logging
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# M5: Basic Monitoring & Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model (using a try-except so CI passes even if training didn't run yet)
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
    logger.info(f"Prediction requested for file: {file.filename}")
    image = Image.open(io.BytesIO(await file.read())).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if model:
        pred = model.predict(img_array)
        class_label = "Dog" if pred[0][0] > 0.5 else "Cat"
        return {"label": class_label, "probability": float(pred[0][0])}
    return {"error": "Model not found. Run train.py first."}