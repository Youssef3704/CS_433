"""Data preparation and feature construction stubs. Fit transformations on training data only."""


def fit_preprocessor(x_train):
    """Return preprocessing state, including learned imputations, scaling and feature selection."""
    pass


def transform_features(x, state):
    """Apply the fitted preprocessing state and return the transformed features."""
    pass


def add_intercept(x):
    """Return the design matrix with a column of ones."""
    pass


def encode_labels(y):
    """Convert dataset labels {-1, 1} to logistic labels {0, 1}."""
    pass
