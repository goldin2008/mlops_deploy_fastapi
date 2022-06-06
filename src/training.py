"""
Author: Ibrahim Sherif
Date: October, 2021
This script used for training, evaluting and saving the model
"""
import sys
import joblib
import logging
from sklearn.model_selection import train_test_split

import config
from pipeline.train import train
from pipeline.evaluate import evaluate
from pipeline.data import get_clean_data
from pipeline.slicing import evaluate_slices


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def run():
    """
    Main entry point
    """
    logging.info("Loading and getting clean data")
    X, y = get_clean_data(config.DATA_DIR)

    logging.info("Splitting data to train and test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)

    logging.info("Started model training")
    model_pipe = train(
        config.MODEL,
        X_train,
        y_train,
        config.PARAM_GRID,
        config.FEATURES)

    logging.info("Evaluating and saving metrics to file")
    with open(config.EVAL_DIR, 'w') as file:
        evaluate(file, model_pipe, X_train, y_train, "train")
        evaluate(file, model_pipe, X_test, y_test, "test")

    logging.info("Evaluating slices and saving to file")
    with open(config.SLICE_DIR, 'w') as file:
        for col in config.SLICE_COLUMNS:
            evaluate_slices(file, model_pipe, col, X_train, y_train, "train")
            evaluate_slices(file, model_pipe, col, X_test, y_test, "test")

    logging.info("Saving model")
    joblib.dump(model_pipe, config.MODEL_DIR)


if __name__ == "__main__":
    run()
