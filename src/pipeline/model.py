"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the model functions needed to build, train and evaluate the model
"""
import sys
import logging
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_model_pipeline(model, feats):
    """
    Creates model pipeline with feature preprocessing steps for
    encoding, scaling and handling missing data

    Args:
        model (sklearn model): sklearn model either RandomForestClassifier/LogisticRegression
        feats (dict): dict of features for each step of the pipeline check config.py

    Returns:
        model_pipe (sklearn pipeline/model): sklearn model or pipeline
    """
    try:
        assert isinstance(model, (LogisticRegression, RandomForestClassifier))
    except AssertionError as error:
        logging.error(
            "Model should be RandomForestClassifier or LogisticRegression %s",
            error)

    if isinstance(model, RandomForestClassifier):
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=1000)
    elif isinstance(model, LogisticRegression):
        encoder = OneHotEncoder(handle_unknown='ignore')

    # categorical feature preprocessor
    categ_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        encoder
    )

    # numerical feature preprocessor
    numeric_preproc = StandardScaler()

    # features preprocessor
    feats_preproc = ColumnTransformer([
        ('drop', 'drop', feats['drop']),
        ('categorical', categ_preproc, feats['categorical']),
        ('numerical', numeric_preproc, feats['numeric'])
    ],
        remainder='passthrough'
    )

    # model pipeline
    model_pipe = Pipeline([
        ('features_preprocessor', feats_preproc),
        ('model', model)
    ])

    return model_pipe


def train_model(model, X_train, y_train, param_grid):
    """
    Performs gridsearch on a model to choose best parameters
    and returns best model found

    Args:
        model (sklearn pipeline/model): sklearn model or pipeline
        X_train (pandas dataframe): Train features data
        y_train (pandas dataframe): Train labels data
        param_grid (dict): Parameters grid check config.py

    Returns:
        model (sklearn pipeline/model): sklearn model or pipeline
    """
    g_search = GridSearchCV(
        model,
        param_grid,
        scoring='f1',
        cv=StratifiedKFold(),
        error_score='raise',
        n_jobs=4
    )

    _ = g_search.fit(X_train, y_train)

    return g_search.best_estimator_


def inference_model(model, X):
    """
    Performs model inference

    Args:
        model (sklearn pipeline/model): sklearn model or pipeline
        X (pandas dataframe): Features data

    Returns:
        None
    """
    preds = model.predict(X)
    return preds
