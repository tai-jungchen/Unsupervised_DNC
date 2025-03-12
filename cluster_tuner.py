import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
import pandas as pd


def fisher_criterion(X, y):
    """
    Compute the Fisher's Criterion for multi-class data and return the optimal projection weights.

    Parameters:
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
    y (numpy.ndarray): Class labels of shape (n_samples,)

    Returns:
    float: Fisher's Criterion value
    numpy.ndarray: Projection weight matrix (n_features, n_classes - 1)
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    overall_mean = np.mean(X, axis=0)

    # Initialize scatter matrices
    S_B = np.zeros((n_features, n_features))
    S_W = np.zeros((n_features, n_features))

    for c in classes:
        X_c = X[y == c]
        mean_c = np.mean(X_c, axis=0)

        # Compute between-class scatter
        n_c = X_c.shape[0]
        mean_diff = (mean_c - overall_mean).reshape(-1, 1)
        S_B += n_c * (mean_diff @ mean_diff.T)

        # Compute within-class scatter
        S_W += np.cov(X_c, rowvar=False) * (n_c - 1)

    # Compute Fisher's criterion as trace(S_B S_W^-1)
    try:
        S_W_inv = np.linalg.pinv(S_W)  # Use pseudo-inverse for stability
        J = np.trace(S_B @ S_W_inv)

        # Compute projection weights (generalized eigenvectors of S_W^-1 S_B)
        eigvals, eigvecs = np.linalg.eig(S_W_inv @ S_B)

        # Sort eigenvectors by descending eigenvalues
        sorted_indices = np.argsort(eigvals)[::-1]
        W = eigvecs[:, sorted_indices[:len(classes) - 1]]  # Select top (C-1) components
    except np.linalg.LinAlgError:
        J = 0  # If singular, assign zero
        W = np.zeros((n_features, len(classes) - 1))

    return J, W


def optimal_k_fisher(X, k_range):
    """
    Determine the optimal number of clusters using Fisher's criterion.

    Parameters:
    - X: ndarray of shape (n_samples, n_features), the data
    - k_range: list or range, possible values of k to evaluate

    Returns:
    - best_k: int, the optimal number of clusters
    - scores: dict, fisher scores for each k
    """
    scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        fisher_score, w = fisher_criterion(X, labels)
        scores[k] = fisher_score

    best_k = max(scores, key=scores.get)  # Highest Fisher score
    return best_k, scores


# Example Usage
if __name__ == "__main__":
    # Generate synthetic data
    df = pd.read_csv("datasets/preprocessed/mpmc.csv")
    df = df[df['failure.type'] != 5]
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['target'], test_size=0.3,
                                                        stratify=df['failure.type'], random_state=0)

    # Find the best k using Fisher's criterion
    k_range = range(2, 10)  # Test k from 2 to 9
    best_k, scores = optimal_k_fisher(X_train.to_numpy(), k_range)

    print(f"Optimal number of clusters (k) based on Fisher's criterion: {best_k}")

    # Plot Fisher scores
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Fisher's Criterion Score")
    plt.title("Optimal k Selection using Fisher's Criterion")
    plt.show()
