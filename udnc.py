"""
Author: Alex (Tai-Jung) Chen

This code implements the proposed UDNC method, an unsupervised learning aided DNC. It utilizes clustering algorithms
to create subclass label, and use OvO to train on these labels.
"""
import numpy as np
import pandas as pd
from imblearn.metrics import specificity_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


def udnc(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                     y_test: pd.DataFrame, maj_clus_algo: object, min_clus_algo: object, verbose: bool = False) -> (
        pd.DataFrame):
    """
    Carry out the DNC plus method on the Machine Predictive Maintenance Classification dataset. The classification
    results will be stored to a .csv file and the console information will be store to a .txt file.

    :param model: classifier.
    :param X_train: training data.
    :param X_test: testing data.
    :param y_train: training label (binary).
    :param y_test: testing label (binary).
    :param maj_clus_algo: clustering algorithm for majority class
    :param min_clus_algo: clustering algorithm for minority class
    :param verbose: whether to print out the confusion matrix or not.
    :return: the dataframe with the classification metrics.
    """
    record_metrics = ['model', 'method', 'f1', 'precision', 'recall', 'bacc', 'kappa', 'acc', 'specificity']
    metrics = {key: [] for key in record_metrics}

    # get multi-class label
    y_train_multi, maj_idx = cluster(X_train, y_train, maj_clus_algo, min_clus_algo)

    # training
    multi_model = OneVsOneClassifier(model)
    multi_model.fit(X_train, y_train_multi)

    # testing
    y_pred_multi = multi_model.predict(X_test)
    y_pred = np.where(y_pred_multi >= maj_idx, 1, 0)

    if verbose:
        print(f'UDNC {model}')
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(classification_report(y_test, y_pred))

    # Store performance
    metrics['acc'].append(round(accuracy_score(y_test, y_pred), 4))
    metrics['kappa'].append(round(cohen_kappa_score(y_test, y_pred), 4))
    metrics['bacc'].append(round(balanced_accuracy_score(y_test, y_pred), 4))
    metrics['precision'].append(round(precision_score(y_test, y_pred), 4))
    metrics['recall'].append(round(recall_score(y_test, y_pred), 4))
    metrics['specificity'].append(round(specificity_score(y_test, y_pred), 4))
    metrics['f1'].append(round(f1_score(y_test, y_pred), 4))

    metrics['model'].append(model)
    metrics['method'].append(f"UDNC_{str(maj_clus_algo).split("(")[0]}")

    return pd.DataFrame(metrics)


def cluster(X, y, maj_clus, min_clus):
    """
    This function applies the clustering algorithm on the minority class to generate minority subclasses.

    :param X: Features.
    :param y: Binary labels.
    :param maj_clus: The clustering algorithm for majority class.
    :param min_clus: The clustering algorithm for minority class.
    :return: The multiclass label after applying clustering.
    """
    X_maj = X[y == 0]
    X_mino = X[y == 1]
    y_multi = y.copy()

    y_maj_multi_label = maj_clus.fit_predict(X_maj)
    y_min_multi_label = min_clus.fit_predict(X_mino) + max(y_maj_multi_label) + 1

    y_multi[y == 0] = y_maj_multi_label
    y_multi[y == 1] = y_min_multi_label
    return y_multi, max(y_maj_multi_label) + 1
