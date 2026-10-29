"""Six required ML methods. Vectors use shape (N,), weights shape (D,)."""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Run full-batch gradient descent and return the final (w, MSE).

    MSE is sum of squared errors / (2*N). The initial weights are not
    modified; zero iterations returns their copy and its corresponding loss.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        w -= gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Run SGD with one randomly sampled observation per update.

    Samples are drawn uniformly with replacement. Set np.random.seed before
    calling for reproducibility. Return the final weights and full-data MSE,
    leaving initial_w unchanged.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        index = np.random.randint(len(y))
        # A one-row slice preserves the batch dimension for the shared gradient.
        gradient = compute_mse_gradient(
            y[index:index + 1], tx[index:index + 1], w
        )
        w -= gamma * gradient
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Return (w, loss) using normal equations; do not use numpy.linalg.lstsq."""
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    return w,compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Return (w, loss) using regularized normal equations; returned MSE excludes penalty."""
    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y 
    w = np.linalg.solve(A, b)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Return (w, loss) using gradient descent; labels must be in {0, 1}."""
    w=np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y,tx,w)
        w = w - gamma * gradient
    return w, compute_logistic_loss(y, tx, w)



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Return (w, loss) with L2 penalty lambda_ * ||w||^2; returned loss excludes penalty."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, compute_logistic_loss(y, tx, w)


def compute_mse(y, tx, w):
    """Return the mean squared error with factor 0.5."""
    residuals = tx @ w - y
    return float(np.mean(residuals ** 2) / 2)


def compute_mse_gradient(y, tx, w):
    """Return the gradient of the mean squared error."""
    return tx.T @ (tx @ w - y) / len(y)


def sigmoid(z):
    """Return the sigmoid of z using a numerically stable calculation."""
    # 1 / (1 + exp(-z)) overflows for large negative z; log(1 + exp(-z)) via
    # logaddexp does not, and exp(-that) is the same value.
    return np.exp(-np.logaddexp(0, -z))


def compute_logistic_loss(y, tx, w):
    """Return the unregularized logistic loss using stable numerical operations."""
    p=tx@w
    return (1/len(y)) * (p *(-y) + np.logaddexp(0,p)).sum(axis=0)


def compute_logistic_gradient(y, tx, w):
    """Return the gradient of the unregularized logistic loss."""
    # divide the D-vector at the end: `(1/len(y)) * tx.T` would bind first and copy
    # the whole design matrix once per iteration.
    return tx.T @ (sigmoid(tx@w) - y) / len(y)
