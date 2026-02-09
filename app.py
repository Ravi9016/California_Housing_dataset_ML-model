#Library Import
import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np

# create the app object
app = FastAPI()
model = pickle.load(open('task1_ml_linear_regression.pkl', 'rb'))

# Index Route, open automatically on the http://127.0.0.1:8000
@app.get("/")
def home():
    return{
        "Message":"Welcome to California Housing Prediction Model"
    }

# Expose the prediction functionality, make a prediction from the passed json data and return the
# predicted value
@app.post("/Predict_Housing_Value")
def Predict_Housing_Value(MedInc: float,
                          HouseAge: float,
                          AveRooms: float,
                          AveBedrms: float,
                          Population: float,
                          AveOccup: float,
                          Latitude: float,
                          Longitude: float):
    
    # 1. Prepare Input
    input_data = np.array([MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude])

    # 2. Get the raw prediction
    input_data_2d = input_data.reshape(1, -1)
    predict = model.predict(input_data_2d)[0]
    Predict_Housing_Value = float(predict)

    # 3. Apply "Logic Explainer" (Post-Prediction Analysis)
    reasons = []
    
    # Income Logic
    if MedInc > 5.0:
        reasons.append("High Median Income is driving the price upward.")
    
    # Geographic/Coastal Logic (Longitude approx -122 is Bay Area/Coast)
    elif -123.0 <= Longitude <= -121.0:
        reasons.append("Proximity to the California coast/Bay Area adds a premium.")
    
    # Density Logic
    elif Population > 5000:
        reasons.append("High population density in this block may negatively impact price.")

    return{
        'MedInc': MedInc,
        'HouseAge' : HouseAge,
        'AveRooms' : AveRooms,
        'AveBedrms' : AveBedrms,
        'Population' : Population,
        'AveOccup' : AveOccup,
        'Latitude' : Latitude,
        'Longitude' : Longitude,
        "Predicted Housing Value" : round(Predict_Housing_Value,4)
    }


 #  ask yourself space and time complexitiy of your code wether it is a psuedo code .