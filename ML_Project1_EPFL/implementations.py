"""Six required ML methods. Vectors use shape (N,), weights shape (D,)."""


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Return (w, loss) using gradient descent; MSE is sum of squared errors / (2*N)."""
    pass


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Return (w, loss) using SGD with batch size 1; use the same MSE convention."""
    pass


def least_squares(y, tx):
    """Return (w, loss) using normal equations; do not use numpy.linalg.lstsq."""
    pass


def ridge_regression(y, tx, lambda_):
    """Return (w, loss) using regularized normal equations; returned MSE excludes penalty."""
    pass


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Return (w, loss) using gradient descent; labels must be in {0, 1}."""
    pass


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Return (w, loss) with L2 penalty lambda_ * ||w||^2; returned loss excludes penalty."""
    pass


def compute_mse(y, tx, w):
    """Return the mean squared error with factor 0.5."""
    pass


def compute_mse_gradient(y, tx, w):
    """Return the gradient of the mean squared error."""
    pass


def sigmoid(z):
    """Return the sigmoid of z using a numerically stable calculation."""
    pass


def compute_logistic_loss(y, tx, w):
    """Return the unregularized logistic loss using stable numerical operations."""
    pass


def compute_logistic_gradient(y, tx, w):
    """Return the gradient of the unregularized logistic loss."""
    pass
