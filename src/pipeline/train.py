"""
Author: Ibrahim Sherif
Date: October, 2021
This script used to train and save the model
"""
import sys
import logging
from pipeline.model import get_model_pipeline, train_model


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def train(model, X_train, y_train, param_grid, feats):
    """
    Creates and trains a pipline for the given model

    Args:
        model (sklearn model): sklearn model
        X_train (pandas dataframe): Train features data
        y_train (pandas dataframe): Train labels data
        param_grid (dict): Parameters grid check config.py
        feats (dict): dict of features for each step of the pipeline check config.py

    Returns:
        model_pipe (sklearn pipeline/model): trained sklearn model or pipeline
    """
    logging.info("Creating model pipeline")
    model_pipe = get_model_pipeline(model, feats)

    logging.info(f"Training {model.__class__.__name__} model")
    model_pipe = train_model(model_pipe, X_train, y_train, param_grid)

    return model_pipe
