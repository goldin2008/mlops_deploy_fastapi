"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the test functions for training model
"""
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

import config
from pipeline.data import get_clean_data
from pipeline.evaluate import compute_metrics
from pipeline.model import get_model_pipeline, train_model, inference_model


def test_model_pipeline():
    """
    Tests if model pipeline is created with correct steps
    """
    feats = config.FEATURES

    model_pipe = get_model_pipeline(config.MODEL, feats)

    assert model_pipe[0].transformers[0][1] == 'drop', "First step: column transformer != drop"
    assert model_pipe[0].transformers[0][2] == feats[
        'drop'], f"{feats['drop']} only should be dropped"

    assert isinstance(model_pipe[0].transformers[1][1],
                      Pipeline), "Second step: column transformer != Pipeline"
    assert isinstance(model_pipe[0].transformers[1][1][0],
                      SimpleImputer), "Second step: column transformer Pipeline != first step: SimpleImputer"
    assert (
        isinstance(model_pipe[1], LogisticRegression) & isinstance(model_pipe[0].transformers[1][1][1], OneHotEncoder)
    ) or (
        isinstance(model_pipe[1], RandomForestClassifier) & isinstance(model_pipe[0].transformers[1][1][1], OrdinalEncoder)
    ), "Second step: column transformer Pipeline != second step: OHE for LogisticRegression or LE for RandomForestClassifier"
    assert model_pipe[0].transformers[1][2] == feats[
        'categorical'], f"{feats['categorical']} only should be included in column transformer second step"

    assert isinstance(model_pipe[0].transformers[2][1],
                      StandardScaler), "Third step of column transformer should be a StandardScaler"
    assert model_pipe[0].transformers[2][2] == feats[
        'numeric'], f"{feats['numeric']} only should be included in column transformer third step"


def test_model_output_shape(sample_data: pd.DataFrame):
    """
    Test model predictions are of correct shape

    Args:
        sample_data (pd.DataFrame): Sample data to be tested
    """
    X_train, X_test, y_train, _ = sample_data
    model = get_model_pipeline(config.MODEL, config.FEATURES)

    model = train_model(model, X_train, y_train, {})

    y_train_pred = inference_model(model, X_train)
    y_test_pred = inference_model(model, X_test)

    assert X_train.shape[
        1] == 14, f"Train data number of columns should be 14 not {X_train.shape[1]}"
    assert X_test.shape[
        1] == 14, f"Test data number of columns should be 14 not {X_test.shape[1]}"
    assert y_train_pred.shape[0] == X_train.shape[
        0], f"Predictions output shape {y_train_pred.shape[0]} is incorrect does not match input shape {X_train.shape[0]}"
    assert y_test_pred.shape[0] == X_test.shape[
        0], f"Predictions output shape {y_test_pred.shape[0]} is incorrect does not match input shape {X_test.shape[0]}"


def test_model_output_range(sample_data: pd.DataFrame):
    """
    Test model predictions are within range 0-1

    Args:
        sample_data (pd.DataFrame): [description]
    """
    X_train, X_test, y_train, _ = sample_data
    model = get_model_pipeline(config.MODEL, config.FEATURES)

    model = train_model(model, X_train, y_train, {})

    y_train_pred = inference_model(model, X_train)
    y_test_pred = inference_model(model, X_test)

    assert (y_train_pred >= 0).all() & (y_train_pred <=
                                        1).all(), "Predictions output range is not from 0-1"
    assert (y_test_pred >= 0).all() & (y_test_pred <= 1).all(
    ), "Predictions output range is not from 0-1"


def test_model_evaluation():
    """
    Test evaluated model metrics are above certain thresholds
    """
    X, y = get_clean_data(config.DATA_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.RANDOM_STATE, stratify=y)

    model = joblib.load(config.MODEL_DIR)

    y_train_pred = inference_model(model, X_train)
    y_test_pred = inference_model(model, X_test)

    pre_train, rec_train, f1_train = compute_metrics(y_train_pred, y_train)
    pre_test, rec_test, f1_test = compute_metrics(y_test_pred, y_test)

    assert pre_train > 0.7, "Train precision should be above 0.85"
    assert rec_train > 0.85, "Train recall should be above 0.85"
    assert f1_train > 0.58, "Train f1 should be above 0.85"

    assert pre_test > 0.68, "Test precision should be above 0.82"
    assert rec_test > 0.82, "Test recall should be above 0.82"
    assert f1_test > 0.56, "Test f1 should be above 0.80"
