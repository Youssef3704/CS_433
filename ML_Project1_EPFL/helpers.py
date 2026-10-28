"""Helper stubs; these are placeholders, not the official EPFL helper implementation."""
import numpy as np
import os
import csv

from ML_Project1_EPFL.implementations import sigmoid


def load_csv_data(data_path, sub_sample=False):
    """Load x_train.csv, x_test.csv and y_train.csv (cached as .npz after the first load)."""
    cache = os.path.join(data_path, "cache_sub.npz" if sub_sample else "cache.npz")

    # fast path: cache already exists
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x_train"], d["x_test"], d["y_train"], d["train_ids"], d["test_ids"]

    # slow path: parse the CSVs once
    y_train = np.genfromtxt(os.path.join(data_path, "y_train.csv"),
                            delimiter=",", skip_header=1, dtype=int, usecols=1)
    x_train = np.genfromtxt(os.path.join(data_path, "x_train.csv"),
                            delimiter=",", skip_header=1)
    x_test = np.genfromtxt(os.path.join(data_path, "x_test.csv"),
                           delimiter=",", skip_header=1)

    train_ids = x_train[:, 0].astype(int)
    test_ids = x_test[:, 0].astype(int)
    x_train, x_test = x_train[:, 1:], x_test[:, 1:]

    if sub_sample:
        y_train, x_train, train_ids = y_train[::50], x_train[::50], train_ids[::50]

    np.savez(cache, x_train=x_train, x_test=x_test, y_train=y_train,
             train_ids=train_ids, test_ids=test_ids)
    return x_train, x_test, y_train, train_ids, test_ids


def load_feature_names(data_path):
    """Return the x_train column names, without the leading Id column."""
    with open(os.path.join(data_path, "x_train.csv")) as f:
        return np.array(next(csv.reader(f))[1:])


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Yield batches of labels and features; SGD must use batch_size=1."""
    indices = np.random.permutation(len(y))



def predict_labels_logistic(weights, data, threshold = 0.5):
    """Return {-1, 1} labels from linear scores."""
    return np.where(sigmoid(data@weights) >= threshold, 1, -1)
