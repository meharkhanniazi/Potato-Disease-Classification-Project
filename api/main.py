from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import tensorflow as tf

app = FastAPI()
endPoint = "http://localhost:8502/v1/models/potatoes_model:predict"
classNames = ['Early Blight', 'Healthy', 'Late Blight']

@app.get("/ping")
async def ping():
    return "I'm alive."

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endPoint, json=json_data)
    prediction = np.array(response.json()['predictions'][0])
    predicted_class = classNames[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        "Class": predicted_class,
        "Confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)