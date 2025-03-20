"""
Author: Alex (Tai-Jung) Chen

This code implements the lda aided DNC method. SMOTE is also applied before LDA dimension reduction.
"""
import numpy as np
import pandas as pd
from imblearn.metrics import specificity_score
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsOneClassifier


class LDAAidedDNC:
    def __init__(self, model: object, smotes: list, metric: str, clus_algo):
        self.model = model
        self.smotes = smotes
        self.metric = metric
        self.clus_algo = clus_algo

    def fit(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
        """
        fit the LDA-aided DNC method.

        :param X_train: training feature
        :param X_test: testing feature
        :param y_train: training labels (binary)
        :param y_test: testing labels (binary)

        :return optimal result based on the metric given.
        """
        res_df = pd.DataFrame()

        # get pseudo-multiclass labels
        y_train_multi, maj_idx = self.cluster(X_train, y_train)

        # run models
        if len(self.smotes) == 0:    # no smote involved
            res = self._algo(X_train.copy(), X_test.copy(), y_train_multi.copy(), y_test.copy(), maj_idx)
            res_df = pd.concat([res_df, res], axis=0)
        else:
            for smote in self.smotes:
                res = self._algo(X_train.copy(), X_test.copy(), y_train_multi.copy(), y_test.copy(), maj_idx, smote=smote)
                res_df = pd.concat([res_df, res], axis=0)

        # return the optimal result in terms of the metric
        opt = res_df[res_df[self.metric] == res_df[self.metric].max()]
        # print(f"Opt smote: {opt['smote']}")
        return opt.drop(columns=["smote"])

    def _algo(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, maj_id: int,
              smote: object = None, verbose: bool = False, discriminate_analysis="LDA"):
        """
        Carry out the lda-aided DNC method.

        :param X_train: training data.
        :param X_test: testing data.
        :param y_train: training label. (multi-class)
        :param y_test: testing label. (binary)
        :param smote: SMOTE object.
        :param verbose: whether to print out the confusion matrix or not.
        :param discriminate_analysis: type of LDA used.

        :return: dataframe with the classification metrics.
        """
        record_metrics = ['model', 'method', 'smote', 'f1', 'precision', 'recall', 'kappa', 'bacc', 'acc',
                          'specificity']
        metrics = {key: [] for key in record_metrics}

        if smote:
            X_smote, y_smote = smote.fit_resample(X_train, y_train)
        else:
            X_smote, y_smote = X_train.copy(), y_train.copy()

        # Fit LDA to training data and transform both train and test sets
        if discriminate_analysis == "LDA":
            da = LDA(n_components=min(y_smote.nunique() - 1, X_smote.shape[1]))
        elif discriminate_analysis == "RDA":
            da = LDA(n_components=y_smote.nunique() - 1, solver='eigen', shrinkage='auto')
        # elif discriminate_analysis == "KDA":
        #     da = KDA(n_components=y_smote.nunique() - 1)
        else:
            raise Exception("Invalid Discriminative Analysis type.")
        X_train_da = da.fit_transform(X_smote, y_smote)
        X_test_da = da.transform(X_test)

        # training
        multi_model = OneVsOneClassifier(self.model)
        multi_model.fit(X_train_da, y_smote)

        # testing
        y_pred_multi = multi_model.predict(X_test_da)
        y_pred = np.where(y_pred_multi >= maj_id, 1, 0)

        if verbose:
            print(f'LDA aided with SMOTE {self.model}') if smote else print(f'LDA aided {self.model}')
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
        metrics['model'].append(self.model)

        if smote:
            smote_name = str(smote).split("(")[0]
            metrics['method'].append(f"UOvO_{str(self.clus_algo)}_{discriminate_analysis}_{smote_name}")
            metrics['smote'].append(smote)
        else:
            metrics['method'].append(f"UOvO_{str(self.clus_algo)}_{discriminate_analysis}")
            metrics['smote'].append(None)
        return pd.DataFrame(metrics)

    def cluster(self, X, y):
        """
        This function applies the clustering algorithm on the minority class to generate minority subclasses.

        :param X: Features.
        :param y: Binary labels.
        :return: The multiclass label after applying clustering.
        """
        # X_maj = X[y == 0]
        X_mino = X[y == 1]
        y_multi = y.copy()

        # y_maj_multi_label = self.maj_clus_algo.fit_predict(X_maj)
        y_min_multi_label = self.clus_algo.fit_predict(X_mino) + 1

        # y_multi[y == 0] = y_maj_multi_label
        y_multi[y == 1] = y_min_multi_label
        return y_multi, 1
