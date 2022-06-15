"""
App main file
"""
# Put the code for your API here.
import os
import sys
import yaml

# Import Uvicorn & the necessary modules from FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field

# from schema import ModelInput
from pandas import DataFrame
import joblib
import starter.ml.data
# Import the PyCaret Regression module
# import pycaret.regression as pycr


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        sys.exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

with open('config.yml') as f:
    config = yaml.load(f)

output_model_path = os.path.join(config['models']['filepath'])

# Initialize the FastAPI application
app = FastAPI()

cat_features = [
"workclass",
"education",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"native-country",
]

class CleanData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=12345)
    education: str = Field(..., example="Masters")
    education_num: int = Field(..., example=14)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Asian-Pac-Islander")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=10)
    capital_loss: int = Field(..., example=10)
    hours_per_week: int = Field(..., example=80)
    native_country: str = Field(..., example="United-States")

# Create a class to store the deployed model & use it for prediction
# class Model:
#     def __init__(self, modelname, bucketname):
#         """
#         To initalize the model
#         modelname: Name of the model stored in the S3 bucket
#         bucketname: Name of the S3 bucket
#         """
#         # Load the deployed model from Amazon S3
#         self.model = pycr.load_model(
#             modelname,
#             platform = 'aws',
#             authentication = { 'bucket' : bucketname }
#         )

#     def predict(self, data):
#         """
#         To use the loaded model to make predictions on the data
#         data: Pandas DataFrame to perform predictions
#         """
#         # Return the column containing the predictions  
#         # (i.e. 'Label') after converting it to a list
#         predictions = pycr.predict_model(self.model, data=data).Label.to_list()
#         return predictions
# # Load the model that you had deployed earlier on S3.
# # Enter your respective bucket name in place of 'mlopsdvc170100035'
# model = Model("lightgbm_deploy_1", "mlopsdvc170100035")


@app.get("/")
async def greetings():
    return {"message": "Udacity MLOps Greetings!"}

# Create the POST endpoint with path '/predict'
@app.post("/predict")
async def inference(input_data: CleanData):
    model = joblib.load("./models/trainedmodel.joblib")
    encoder = joblib.load("./models/encoder.joblib")
    lb = joblib.load("./models/lb.joblib")

    input_df = DataFrame(data=input_data.values(), index=input_data.keys()).T
    X, _, _, _ = starter.ml.data.process_data(input_df,
                                              categorical_features=cat_features,
                                              encoder=encoder, lb=lb, training=False)
    prediction = inference(model, X)
    y_pred = lb.inverse_transform(prediction)[0]

    return {"prediction": y_pred}
