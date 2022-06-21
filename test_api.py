"""
This script holds the test functions for api module
"""
import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_greetings():
    """
    Tests GET greetings function
    """
    response = client.get('/')
    # print(response)
    # print(response.status_code)
    # print(response.json())
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "GET"
    assert response.json() == {'message': 'Udacity MLOps Greetings!'}

def test_predict_gr50k():
    r = client.post("/predict", json={
        "age": 39,
        "workclass": "Private",
        "fnlgt": 12345,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital_gain": 10,
        "capital_loss": 10,
        "hours_per_week": 80,
        "native_country": "United-States"
    })
    # print(r)
    # print(r.status_code)
    # print(r.json())
    # print(r.request.method)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_predict_le50k():
    r = client.post("/predict", json={
        "age": 53,
        "workclass": "Private",
        "fnlgt": 234721,
        "education": "11th",
        "education_num": 7,
        "marital_status": "Married-civ-spouse",
        "occupation": "Handlers-cleaners",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })
    # print(r)
    # print(r.status_code)
    # print(r.json())
    # print(r.request.method)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_predict_status():
    """
    Tests POST predict function status
    """
    data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 12345,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital_gain": 10,
        "capital_loss": 10,
        "hours_per_week": 80,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=data)
    # print(r)
    # print(r.status_code)
    # print(r.request.method)
    assert response.status_code == 200
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "POST"


def test_missing_feature_predict():
    """
    Tests POST predict function when failed due to missing features
    """
    data = {
        "age": 0,
        'fnlgt': 15,
        'education_num': 1,
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 5
    }
    response = client.post("/predict", json=data)
    # print(response)
    # print(response.status_code)
    # print(response.request.method)
    # print(response.json())
    # print(HTTPStatus.UNPROCESSABLE_ENTITY)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.request.method == "POST"
    assert response.json()["detail"][0]["type"] == "value_error.missing"

if __name__ == "__main__":
    test_greetings()
    test_predict_gr50k()
    test_predict_le50k()
    test_predict_status()
    test_missing_feature_predict()