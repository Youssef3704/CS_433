"""Data preparation and feature construction stubs. Fit transformations on training data only."""
import numpy as np

def fit_preprocessor(x_train):
    """Return preprocessing state, including learned imputations, scaling and feature selection."""
    x = np.asanyarray(x_train, dtype=np.float64)
    keep = np.isnan(x).mean(axis=0) <= 0.5
    x = x[:, keep]
    median=np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), median, x)
    mean, std = x.mean(axis=0), x.std(axis=0)
    nonconst = std > 0
    return {
        "keep_idx": np.flatnonzero(keep)[nonconst],
        "median": median[nonconst],
        "mean": mean[nonconst],
        "std": std[nonconst],
    }

def transform_features(x, state):
    """Apply the fitted preprocessing state and return the transformed features."""
    x = np.asarray(x, dtype=np.float64)[:, state["keep_idx"]]
    x = np.where(np.isnan(x), state["median"], x)
    return add_intercept((x - state["mean"]) / state["std"])


def add_intercept(x):
    """Return the design matrix with a column of ones."""
    ones = np.ones(shape=(x.shape[0], 1))
    return np.concatenate((ones, x), axis=1)


def encode_labels(y):
    """Convert dataset labels {-1, 1} to logistic labels {0, 1}."""
    return np.where(y == 1, 1, 0)
