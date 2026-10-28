"""Local evaluation and model selection stubs."""
import numpy as np

def split_data(x, y, train_ratio=0.8, seed=1):
    """Return x_train, x_valid, y_train, y_valid using a reproducible split."""
    np.random.seed(seed)
    indices = np.random.permutation(len(x))
    x_new = x[indices]
    y_new = y[indices]

    ratio = int(train_ratio*len(y))
    x_train = x_new[:ratio]
    x_valid = x_new[ratio:]
    y_train = y_new[:ratio]
    y_valid = y_new[ratio:]

    return x_train, x_valid, y_train, y_valid


def build_k_indices(y, k_fold, seed=1):
    """Return validation fold indices, including every sample exactly once."""
    np.random.seed(seed)
    indices = np.random.permutation(len(y))
    return np.array_split(indices, k_fold)


def compute_accuracy(y_true, y_pred):
    """Return classification accuracy."""
    return  (y_true == y_pred).sum()/len(y_true)


def compute_f1(y_true, y_pred, positive_label=1):
    """Return the binary F1 score with a defined zero-division convention."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_pos = y_true == positive_label
    pred_pos = y_pred == positive_label

    TP = np.sum(true_pos & pred_pos)
    FP = np.sum(~true_pos & pred_pos)
    FN = np.sum(true_pos & ~pred_pos)

    denom = 2 * TP + FP + FN
    if denom == 0:
        return 0.0  # no positives in truth or predictions
    return float(2 * TP / denom)


def cross_validate(x, y, train_fn, predict_fn=None, k_fold=5, seed=1):
    """Return the per-fold model and indices, plus metrics when predict_fn is given.

    train_fn must fit preprocessing on its own training fold so that no
    validation fold leaks into the fitted transformation. Leaving predict_fn
    out skips the metrics, which lets the caller score the returned models
    several times (over decision thresholds, say) without refitting them.
    """
    indices=build_k_indices(y,k_fold,seed)
    fold={}
    for i in range(k_fold):
        valid_idx = indices[i]
        train_idx = np.concatenate([indices[j] for j in range(k_fold) if j != i])
        training_model=train_fn(x[train_idx], y[train_idx])

        fold[i] = {"model": training_model, "train_idx": train_idx, "valid_idx": valid_idx}
        if predict_fn is None:
            continue

        y_true_train=y[train_idx]
        y_pred_train=predict_fn(training_model, x[train_idx])
        y_true_test=y[valid_idx]
        y_pred_test=predict_fn(training_model, x[valid_idx])

        fold[i].update({"accuracy_train": compute_accuracy(y_true_train, y_pred_train),
                        "f1_score_train": compute_f1(y_true_train, y_pred_train),
                        "accuracy_test": compute_accuracy(y_true_test, y_pred_test),
                        "f1_score_test": compute_f1(y_true_test, y_pred_test)})
    return fold

