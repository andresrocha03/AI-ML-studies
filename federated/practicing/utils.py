import os
import flwr as fl
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
from typing import List
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
import pandas as pd

NUM_UNIQUE_LABELS = 2  # Number of unique labels in your dataset
NUM_FEATURES = 8  # Number of features in your dataset


def set_initial_params(model: LogisticRegression):
    """Set initial parameters for the model."""
    model.classes_ = np.array([i for i in range(NUM_UNIQUE_LABELS)])
    model.coef_ = np.zeros((NUM_UNIQUE_LABELS, NUM_FEATURES))
    if model.fit_intercept:
        model.intercept_ = np.zeros((NUM_UNIQUE_LABELS,))
    return model

def get_model_parameters(model: LogisticRegression):
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_model_params(model: LogisticRegression, params: List[NDArray]):
    """Set model parameters."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def get_scores(y_true: NDArray, y_prob: NDArray) -> dict:
    """
    Get scores.
    Input:
    - y_true: np.array
        True labels.
    - y_prob: np.array
        Predicted probabilities.
    Output:
    - score: dict
        Dictionary containing the loss, accuracy, precision, and recall.

    """
    y_pred = np.argmax(y_prob, axis=1)
    score = {
        "loss": log_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
    }
    return score


def load_data(partition: pd.DataFrame):
    """Load data."""
    X = partition.drop('label', axis=1).values
    y = partition['label'].values
    return X, y 


def partition_data(data, num_partitions):
# Partitioning the dataset into parts for each client
    return np.array_split(data, num_partitions)
