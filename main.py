"""
Author: Alex (Tai-Jung) Chen

Run through all the classification framework for comparison purpose.
"""
from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.mixture import GaussianMixture

from binary_clf import binary
from udnc import udnc


def main(dataset: str, models: List[str], n_rep: int, cluster: object) -> pd.DataFrame:
    """
    Run through all the methods for comparison.

    :param dataset: dataset for testing.
    :param models: types of models used for classification.
    :param n_rep: number of replications.
    :param cluster: clustering algorithm
    :return results stored in the DataFrame.
    """
    res_df = pd.DataFrame()
    scaler = StandardScaler()

    # read data
    for i in tqdm(range(n_rep)):
        if dataset == 'MPMC':
            df = pd.read_csv("datasets/preprocessed/mpmc.csv")
            df = df[df['failure.type'] != 5]
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['target'], test_size=0.3,
                                                                stratify=df['failure.type'], random_state=i)
        elif dataset == 'GLASS':
            df = pd.read_csv("datasets/preprocessed/glass_data.csv")
            df['target'] = np.where(df['Type'].isin([0, 1]), 0, 1)
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['target'], test_size=0.3,
                                                                stratify=df['Type'], random_state=i)
        elif dataset == 'MNIST':
            df = pd.read_csv("datasets/preprocessed/imb_mnist.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['label'], test_size=0.3,
                                                                stratify=df['label'], random_state=i)
        elif dataset == 'USPS':
            df = pd.read_csv("datasets/preprocessed/imb_digit.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.3,
                                                                stratify=df.iloc[:, 0], random_state=i)
        else:
            raise Exception("Invalid dataset.")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # run models
        for model in models:
            res_bin = binary(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(),
                             verbose=True)
            res_udnc = udnc(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(),
                            cluster, verbose=True)
            res_df = pd.concat([res_df, res_bin, res_udnc], axis=0)

    # average the performance
    return res_df.groupby(by=["method", "model"], sort=False).mean()


if __name__ == "__main__":
    N_REP = 10

    ##### MPMC #####
    DATASET = "MPMC"
    MODELS = [LogisticRegression(),
              GaussianNB(),
              LDA(),
              SVC(kernel='linear', C=0.1),
              SVC(kernel='rbf', C=0.5),
              DecisionTreeClassifier(random_state=42),
              RandomForestClassifier(random_state=42),
              GradientBoostingClassifier(random_state=42),
              xgb.XGBClassifier(random_state=42)]

    CLUS = KMeans(n_clusters=3)
    # CLUS = GaussianMixture(n_components=5, covariance_type='full')
    # CLUS = AgglomerativeClustering(n_clusters=2, linkage='ward')
    ##### MPMC #####

    ##### USPS #####
    # DATASET = "USPS"
    # MODELS = [LogisticRegression(max_iter=10000), GaussianNB(), LDA(), SVC(kernel='linear', C=0.1),
    #           SVC(kernel='rbf', C=0.5), DecisionTreeClassifier(), RandomForestClassifier(),
    #           GradientBoostingClassifier(random_state=42), xgb.XGBClassifier()]
    # SMOTE_INST_1 = BorderlineSMOTE(kind='borderline-1')
    ##### USPS #####

    ##### MNIST #####
    # DATASET = "MNIST"
    # MODELS = [LogisticRegression(max_iter=300), GaussianNB(), LDA(), #SVC(kernel='linear', C=0.1),
    #           SVC(kernel='rbf', C=0.5), DecisionTreeClassifier(), RandomForestClassifier(),
    #           GradientBoostingClassifier(random_state=42), xgb.XGBClassifier()]
    # SMOTE_INST_1 = BorderlineSMOTE(kind='borderline-1')
    ##### MNIST #####

    ##### GLASS #####
    # DATASET = "GLASS"
    # MODELS = [LogisticRegression(), GaussianNB(),  SVC(kernel='linear', C=1),
    #           SVC(kernel='rbf', C=1), DecisionTreeClassifier(), RandomForestClassifier(),
    #           GradientBoostingClassifier(), xgb.XGBClassifier()]
    # # CLUS = KMeans(n_clusters=4)
    # CLUS = GaussianMixture(n_components=3, covariance_type='full')
    ##### GLASS #####

    res = main(DATASET, MODELS, N_REP, CLUS)
    filename = f"results_0308_{DATASET}.csv"
    res.to_csv(filename)

