# Put the code for your API here.
import os
import yaml

# Import Uvicorn & the necessary modules from FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

# Import the PyCaret Regression module
import pycaret.regression as pycr

from schema import ModelInput
from pandas import DataFrame
# from starter.inferance_model import run_inference
import joblib
import starter.ml.data


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

with open('config.yml') as f:
    config = yaml.load(f)

output_model_path = os.path.join(config['model']['filepath'])

# Initialize the FastAPI application
app = FastAPI()


# # Create a class to store the deployed model & use it for prediction
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
async def get_items():
    return {"message": "Udacity MLOps Greetings!"}

# Create the POST endpoint with path '/predict'
@app.post("/predict")
async def inference(input_data: ModelInput):
    model = joblib.load("./model/trainedmodel.joblib")
    encoder = joblib.load("./model/encoder.joblib")
    lb = joblib.load("./model/lb.joblib")

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

    input_df = DataFrame(data=input_data.values(), index=input_data.keys()).T
    X, _, _, _ = starter.ml.data.process_data(
                input_df,
                categorical_features=cat_features,
                encoder=encoder, lb=lb, training=False)
    prediction = inference(model, X)
    y_pred = lb.inverse_transform(prediction)[0]

    return {"prediction": y_pred}
