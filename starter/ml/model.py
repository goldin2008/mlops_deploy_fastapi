from sklearn.metrics import fbeta_score, precision_score, recall_score

import logging
from numpy import mean
from numpy import std
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred


def slices(model, cat, X, cat_features, encoder, lb):
    """ Computes performance on model slices
    Inputs
    ------
    model : ???
        Trained machine learning model.
    cat : str
        category to be sliced
    X : np.array
        Data used for prediction.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, for processing data.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, for processing data.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    Returns
    -------
    No returns
    """
    with open("slice_output.txt", "w") as f:
        f.write(cat)
        f.write('\n')
        for val in X[cat].unique():
            X_slice = X[X[cat] == val]
            X_test, y_test, encoder, lb = process_data(
                X_slice, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb=lb
                )
            # print(X.shape)
            # print(X_temp.shape)
            y_pred = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            f.writelines("-------------------------------------------------")
            f.write('\n')
            f.write(val)
            f.write('\n')
            f.write('Precision: ' + str(precision))
            f.write('\n')
            f.write('Recall: ' + str(recall))
            f.write('\n')
            f.write('F-Beta Score: ' + str(fbeta))
            f.write('\n')

    return