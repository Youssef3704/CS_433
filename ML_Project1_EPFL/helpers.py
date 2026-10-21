"""Helper stubs; these are placeholders, not the official EPFL helper implementation."""


def load_csv_data(data_path, sub_sample=False):
    """Load x_train.csv, x_test.csv and y_train.csv; return x_train, x_test, y_train, train_ids, test_ids. Verify against the official helper before implementing."""
    pass


def create_csv_submission(ids, y_pred, name):
    """Write IDs and predicted labels to the competition submission CSV."""
    pass


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Yield batches of labels and features; SGD must use batch_size=1."""
    pass


def predict_labels(weights, data):
    """Return {-1, 1} labels from linear scores."""
    pass
