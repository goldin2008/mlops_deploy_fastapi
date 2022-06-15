import os
import sys
import joblib

from starter.ml.model import slices
import pandas as pd

model = joblib.load("./models/model.joblib")
lb = joblib.load("./models/lb.joblib")
encoder = joblib.load("./models/encoder.joblib")

data = pd.read_csv("data/census_clean.csv")

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

slices(model, "education", data, cat_features, encoder, lb)