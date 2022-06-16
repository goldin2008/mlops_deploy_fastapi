import requests


data = {
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
}

# GET request
response = requests.get('http://127.0.0.1:8000/')
print(response.status_code)
print(response.json())

# POST request
response = requests.post('http://127.0.0.1:8000/predict', auth=('user', 'pass'), json=data)
print(response.status_code)
print(response.json())
