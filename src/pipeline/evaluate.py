import sys
import logging
from sklearn.metrics import fbeta_score, precision_score, recall_score, classification_report

from pipeline.model import inference_model


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def compute_metrics(y_true, y_pred):
    """
    Computes precision, recall and f1 scores

    Args:
        y_true (array): array of true labels
        y_pred (array): array of predicted labels

    Returns:
        f1 (float): f1 score
        precision (float): precision score
        recall (float): recall score
    """
    f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    # print(classification_report(y_true, y_pred))

    return f1, precision, recall


def evaluate(file, model_pipe, X, y, split):
    """
    Evaluates a model againest a specific data split and
    saves it in a file

    Args:
        file (file): file object
        model_pipe (sklearn pipeline/model): sklearn model or pipeline
        X (pandas dataframe): data features
        y (pandas series): data labels
        split (str): train or test split

    Returns:
        Nones
    """
    logging.info("Running inference")
    y_pred = inference_model(model_pipe, X)

    logging.info("Evaluating model")
    pre, rec, f1 = compute_metrics(y_pred, y)

    logging.info(f"Evalating {split} data")
    logging.info("Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}".format(
        pre, rec, f1))

    print(f"Evaluation on {split} data", file=file)
    print(
        f"Precision = {pre:.3f}, Recall = {rec:.3f}, F1 = {f1:.3f}",
        file=file)
