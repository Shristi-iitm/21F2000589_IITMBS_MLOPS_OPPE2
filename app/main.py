from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the input data structure using Pydantic
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Log the input data
    logger.info(f"Received prediction request for data: {data.dict()}")

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].tolist()

    # Log the output
    result = {
        'prediction': int(prediction),
        'probability_no_disease': probability[0],
        'probability_disease': probability[1]
    }
    logger.info(f"Prediction result: {result}")
    
    return result
