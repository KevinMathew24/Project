

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

limit = .70
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("D:/Project/Leaf disease classification/leaf-disease-classification/models/5")

CLASS_NAMES = ["Potato Early Blight", "Potato Late Blight", "Potato Healthy","Tomato Early Blight","Tomato Late Blight","Tomato Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    print("Hello")
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    if(float(confidence) < float(limit)):
        return {
          'class': "Unable to Predict. Low Confidence",
          'confidence': "-"
        }
    else:     
        return {
          'class': predicted_class,
          'confidence': float(confidence)
        }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

