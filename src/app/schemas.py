"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds schemas for fastapi app
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class FeatureInfo(str, Enum):
    age = "age"
    workclass = "workclass"
    fnlgt = "fnlgt"
    education = "education"
    education_num = "education_num"
    marital_status = "marital_status"
    occupation = "occupation"
    relationship = "relationship"
    race = "race"
    sex = "sex"
    captial_gain = "capital_gain"
    captial_loss = "capital_loss"
    hours_per_week = "hours_per_week"
    native_country = "native_country"


class Person(BaseModel):
    age: int
    workclass: Optional[str] = None
    fnlgt: int
    education: Optional[str] = None
    education_num: int
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Optional[str] = None
