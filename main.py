import uvicorn #taking ASGI request
from fastapi import FastAPI
#from Telcoms import Telcom
import numpy as np
import pickle
import pandas as pd
import os


#create app object
app = FastAPI()
pickle_in = open('classifiertelcom.pkl','rb')
classifier = pickle.load(pickle_in)


from pydantic import BaseModel
class Telcom(BaseModel):
    TotalCharges : float
    tenure : float
    SeniorCitizen : float
    MonthlyCharges: float
    Month_to_month: float

@app.get('/')
def index():
    return {'message':"hello world"}

#giving names
@app.get("/{name}")
def get_name(name:str):
    return {'welcome to this':f'{name}'}


@app.post('/predict')
def predict_Telcom(data:Telcom):
    data = data.dict()
    print(data)
    print("Hello")
    TotalCharges = data['TotalCharges']
    tenure = data['tenure']
    SeniorCitizen = data['SeniorCitizen']
    MonthlyCharges = data['MonthlyCharges']
    Monthtomonth = data['Month_to_month']

    
    print(classifier.predict([[TotalCharges,tenure,SeniorCitizen,MonthlyCharges,Monthtomonth]]))
    print("Hello")
    prediction = classifier.predict([[TotalCharges,tenure,SeniorCitizen,MonthlyCharges,Monthtomonth]])
    if(prediction[0]>0.5):
        prediction = 'Churned'
    else:
        prediction = 'not churned'
    return{
        'prediction':prediction
    }


if __name__ == "__main__":
    uvicorn.run(app)