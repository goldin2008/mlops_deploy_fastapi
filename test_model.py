'''
Testing data and model scripts
'''
import os
import sys
import pandas as pd
import logging
import pytest
import joblib
# import yaml
import starter
from starter.ml import data as d
from starter.ml import model as m
from starter import train_model as tm
import unittest


@pytest.fixture
def data():
    """
    Get the cleaned dataset
    """
    # cwd = os.getcwd()
    df = d.import_data("./data/census_clean.csv")
    return df

@pytest.fixture
def model():
    model = joblib.load("./models/model.joblib")
    return model

def test_import_data(data):
    '''
    test import_data 
    '''
    assert data.shape[0] > 0
    assert data.shape[1] > 0
    

def test_process_data(data):
    """
    test process_data function
    """
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

    X_test, y_test, _, _ = d.process_data(
        data, categorical_features=cat_features, label="salary", training=True)
       
    assert len(X_test) == len(y_test)

def test_columns_name(data):
    """
    test the columns dataset name
    """
    expected_columns = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary"
    ]

    obtained_columns = data.columns.values
    assert list(expected_columns) == list(obtained_columns)
    
def test_train_save_model():
    '''
    test train_models
    '''
    # Models
    # 1. Check if the list is empty or not
    # 2. Check if all files exist
    try:
        # Getting the list of directories
        path = "./models"
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        assert os.path.isfile('./models/model.joblib')
        assert os.path.isfile('./models/encoder.joblib')
        assert os.path.isfile('./models/lb.joblib')
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Model files not found")
        raise err

# if __name__ == '__main__':
#     unittest.main()