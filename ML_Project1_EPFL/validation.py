"""Local evaluation and model selection stubs."""


def split_data(x, y, train_ratio=0.8, seed=1):
    """Return x_train, x_valid, y_train, y_valid using a reproducible split."""
    pass


def build_k_indices(y, k_fold, seed=1):
    """Return validation fold indices, including every sample exactly once."""
    pass


def compute_accuracy(y_true, y_pred):
    """Return classification accuracy."""
    pass


def compute_f1(y_true, y_pred, positive_label=1):
    """Return the binary F1 score with a defined zero-division convention."""
    pass


def cross_validate(x, y, train_fn, predict_fn, k_fold=5, seed=1):
    """Return fold metrics; fit preprocessing independently inside each training fold."""
    pass
